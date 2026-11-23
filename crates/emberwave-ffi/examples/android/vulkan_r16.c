/*
 * Verifies emberwave's Vulkan device interop and R16_UNORM decoded plane output
 * against the C API on Android (or any Vulkan system).
 *
 *     emberwave_vulkan_r16 INPUT.pwv1
 *
 * Creates a Vulkan instance and device with the extensions and features
 * reported by emberwave, queries EMBERWAVE_CONTEXT_FEATURE_PLANE_R16, wraps
 * caller-owned VkImages, decodes the first frame via
 * emberwave_decoder_decode_plane_textures, and reads the planes back to verify
 * pixel values (handling unsupported R16 cleanly if the device lacks it).
 */
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#include "emberwave.h"

static int fail_ew(const char *what, EmberwaveStatus status)
{
    fprintf(stderr, "%s failed (status %d): %s\n", what, (int)status, emberwave_last_error());
    return 1;
}

static int fail_vk(const char *what, VkResult result)
{
    fprintf(stderr, "%s failed (VkResult %d)\n", what, (int)result);
    return 1;
}

static unsigned char *read_file(const char *path, size_t *size)
{
    FILE *file = fopen(path, "rb");
    unsigned char *data = NULL;
    size_t capacity = 0;
    int error;

    if (!file) {
        perror(path);
        return NULL;
    }
    *size = 0;
    for (;;) {
        size_t got;
        if (*size == capacity) {
            unsigned char *grown;
            capacity = capacity ? 2 * capacity : (size_t)1 << 20;
            grown = realloc(data, capacity);
            if (!grown) {
                fprintf(stderr, "%s: out of memory\n", path);
                free(data);
                fclose(file);
                return NULL;
            }
            data = grown;
        }
        got = fread(data + *size, 1, capacity - *size, file);
        if (got == 0) {
            break;
        }
        *size += got;
    }
    error = ferror(file);
    fclose(file);
    if (error) {
        perror(path);
        free(data);
        return NULL;
    }
    return data;
}

static size_t read_le32(const unsigned char *bytes)
{
    return (size_t)bytes[0] | ((size_t)bytes[1] << 8) | ((size_t)bytes[2] << 16) |
           ((size_t)bytes[3] << 24);
}

typedef struct VulkanContext {
    VkInstance instance;
    VkPhysicalDevice physical_device;
    VkDevice device;
    uint32_t queue_family;
    VkQueue queue;
    VkCommandPool command_pool;
    VkPhysicalDeviceMemoryProperties mem_props;
} VulkanContext;

typedef struct PlaneImage {
    VkImage image;
    VkDeviceMemory memory;
    uint32_t width;
    uint32_t height;
    uint32_t texel_bytes;
} PlaneImage;

static uint32_t find_memory_type(const VulkanContext *vk,
                                 uint32_t type_bits,
                                 VkMemoryPropertyFlags wanted)
{
    uint32_t i;
    for (i = 0; i < vk->mem_props.memoryTypeCount; i++) {
        if ((type_bits & (1u << i)) &&
            (vk->mem_props.memoryTypes[i].propertyFlags & wanted) == wanted) {
            return i;
        }
    }
    return UINT32_MAX;
}

static int create_plane_image(const VulkanContext *vk,
                              VkFormat format,
                              VkImageUsageFlags usage,
                              uint32_t width,
                              uint32_t height,
                              uint32_t texel_bytes,
                              PlaneImage *out)
{
    VkImageCreateInfo img_info = {0};
    VkMemoryRequirements reqs = {0};
    VkMemoryAllocateInfo alloc_info = {0};
    uint32_t mem_type;
    VkResult res;

    memset(out, 0, sizeof *out);
    out->width = width;
    out->height = height;
    out->texel_bytes = texel_bytes;

    img_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    img_info.imageType = VK_IMAGE_TYPE_2D;
    img_info.format = format;
    img_info.extent.width = width;
    img_info.extent.height = height;
    img_info.extent.depth = 1;
    img_info.mipLevels = 1;
    img_info.arrayLayers = 1;
    img_info.samples = VK_SAMPLE_COUNT_1_BIT;
    img_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    img_info.usage = usage;
    img_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    img_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    res = vkCreateImage(vk->device, &img_info, NULL, &out->image);
    if (res != VK_SUCCESS) {
        return fail_vk("vkCreateImage", res);
    }

    vkGetImageMemoryRequirements(vk->device, out->image, &reqs);
    mem_type = find_memory_type(vk, reqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (mem_type == UINT32_MAX) {
        mem_type = find_memory_type(vk, reqs.memoryTypeBits, 0);
    }
    if (mem_type == UINT32_MAX) {
        fprintf(stderr, "no compatible memory type for plane image\n");
        vkDestroyImage(vk->device, out->image, NULL);
        out->image = VK_NULL_HANDLE;
        return 1;
    }

    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = reqs.size;
    alloc_info.memoryTypeIndex = mem_type;
    res = vkAllocateMemory(vk->device, &alloc_info, NULL, &out->memory);
    if (res != VK_SUCCESS) {
        vkDestroyImage(vk->device, out->image, NULL);
        out->image = VK_NULL_HANDLE;
        return fail_vk("vkAllocateMemory", res);
    }

    res = vkBindImageMemory(vk->device, out->image, out->memory, 0);
    if (res != VK_SUCCESS) {
        vkFreeMemory(vk->device, out->memory, NULL);
        vkDestroyImage(vk->device, out->image, NULL);
        out->memory = VK_NULL_HANDLE;
        out->image = VK_NULL_HANDLE;
        return fail_vk("vkBindImageMemory", res);
    }
    return 0;
}

static void destroy_plane_image(const VulkanContext *vk, PlaneImage *img)
{
    if (img->image != VK_NULL_HANDLE) {
        vkDestroyImage(vk->device, img->image, NULL);
        img->image = VK_NULL_HANDLE;
    }
    if (img->memory != VK_NULL_HANDLE) {
        vkFreeMemory(vk->device, img->memory, NULL);
        img->memory = VK_NULL_HANDLE;
    }
}

static int readback_image(const VulkanContext *vk,
                          VkImage image,
                          uint32_t width,
                          uint32_t height,
                          uint32_t texel_bytes,
                          unsigned char *dst)
{
    VkDeviceSize byte_size = (VkDeviceSize)width * height * texel_bytes;
    VkBufferCreateInfo buf_info = {0};
    VkBuffer buffer = VK_NULL_HANDLE;
    VkMemoryRequirements reqs = {0};
    VkMemoryAllocateInfo alloc_info = {0};
    VkDeviceMemory memory = VK_NULL_HANDLE;
    uint32_t mem_type;
    VkCommandBufferAllocateInfo cmd_alloc = {0};
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkCommandBufferBeginInfo begin_info = {0};
    VkImageMemoryBarrier to_src = {0};
    VkImageMemoryBarrier to_read = {0};
    VkBufferImageCopy region = {0};
    VkSubmitInfo submit = {0};
    VkFenceCreateInfo fence_info = {0};
    VkFence fence = VK_NULL_HANDLE;
    void *mapped = NULL;
    VkResult res;
    int rc = 1;

    buf_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buf_info.size = byte_size;
    buf_info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    buf_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    res = vkCreateBuffer(vk->device, &buf_info, NULL, &buffer);
    if (res != VK_SUCCESS) {
        return fail_vk("vkCreateBuffer", res);
    }

    vkGetBufferMemoryRequirements(vk->device, buffer, &reqs);
    mem_type = find_memory_type(
        vk, reqs.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (mem_type == UINT32_MAX) {
        fprintf(stderr, "no host-coherent memory type for staging buffer\n");
        goto cleanup;
    }

    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = reqs.size;
    alloc_info.memoryTypeIndex = mem_type;
    res = vkAllocateMemory(vk->device, &alloc_info, NULL, &memory);
    if (res != VK_SUCCESS) {
        fail_vk("vkAllocateMemory(staging)", res);
        goto cleanup;
    }
    res = vkBindBufferMemory(vk->device, buffer, memory, 0);
    if (res != VK_SUCCESS) {
        fail_vk("vkBindBufferMemory", res);
        goto cleanup;
    }

    cmd_alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmd_alloc.commandPool = vk->command_pool;
    cmd_alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmd_alloc.commandBufferCount = 1;
    res = vkAllocateCommandBuffers(vk->device, &cmd_alloc, &cmd);
    if (res != VK_SUCCESS) {
        fail_vk("vkAllocateCommandBuffers", res);
        goto cleanup;
    }

    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    res = vkBeginCommandBuffer(cmd, &begin_info);
    if (res != VK_SUCCESS) {
        fail_vk("vkBeginCommandBuffer", res);
        goto cleanup;
    }

    to_src.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    to_src.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    to_src.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    to_src.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    to_src.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    to_src.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_src.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_src.image = image;
    to_src.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    to_src.subresourceRange.levelCount = 1;
    to_src.subresourceRange.layerCount = 1;

    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 0, NULL, 0, NULL, 1, &to_src);

    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.layerCount = 1;
    region.imageExtent.width = width;
    region.imageExtent.height = height;
    region.imageExtent.depth = 1;
    vkCmdCopyImageToBuffer(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffer, 1, &region);

    to_read = to_src;
    to_read.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    to_read.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
    to_read.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    to_read.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         0, 0, NULL, 0, NULL, 1, &to_read);

    res = vkEndCommandBuffer(cmd);
    if (res != VK_SUCCESS) {
        fail_vk("vkEndCommandBuffer", res);
        goto cleanup;
    }

    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    res = vkCreateFence(vk->device, &fence_info, NULL, &fence);
    if (res != VK_SUCCESS) {
        fail_vk("vkCreateFence", res);
        goto cleanup;
    }

    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;
    res = vkQueueSubmit(vk->queue, 1, &submit, fence);
    if (res != VK_SUCCESS) {
        fail_vk("vkQueueSubmit", res);
        goto cleanup;
    }
    res = vkWaitForFences(vk->device, 1, &fence, VK_TRUE, 10000000000ULL);
    if (res != VK_SUCCESS) {
        fail_vk("vkWaitForFences", res);
        goto cleanup;
    }

    res = vkMapMemory(vk->device, memory, 0, byte_size, 0, &mapped);
    if (res != VK_SUCCESS) {
        fail_vk("vkMapMemory", res);
        goto cleanup;
    }
    memcpy(dst, mapped, (size_t)byte_size);
    vkUnmapMemory(vk->device, memory);
    rc = 0;

cleanup:
    if (fence != VK_NULL_HANDLE) {
        vkDestroyFence(vk->device, fence, NULL);
    }
    if (cmd != VK_NULL_HANDLE) {
        vkFreeCommandBuffers(vk->device, vk->command_pool, 1, &cmd);
    }
    if (buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(vk->device, buffer, NULL);
    }
    if (memory != VK_NULL_HANDLE) {
        vkFreeMemory(vk->device, memory, NULL);
    }
    return rc;
}

static uint32_t g_released_count = 0;

static void on_image_release(void *user)
{
    (void)user;
    g_released_count++;
}

int main(int argc, char **argv)
{
    size_t file_size = 0;
    unsigned char *file_bytes = NULL;
    size_t frame_size = 0;
    const unsigned char *frame_bytes = NULL;
    EmberwaveStreamInfo stream = {.struct_size = sizeof stream};
    EmberwaveStatus status;

    uint32_t inst_ext_count = 0;
    const char **inst_ext_names = NULL;
    uint32_t api_version = VK_API_VERSION_1_0;
    PFN_vkEnumerateInstanceVersion enum_inst_ver;
    VkApplicationInfo app_info = {0};
    VkInstanceCreateInfo inst_info = {0};
    VulkanContext vk = {0};
    VkResult vk_res;

    uint32_t phys_count = 0;
    VkPhysicalDevice *phys_devices = NULL;
    VkPhysicalDeviceProperties phys_props = {0};
    EmberwaveVulkanInstance ew_inst = {.struct_size = sizeof ew_inst};
    EmberwaveVulkanRequirements *reqs = NULL;
    uint32_t dev_ext_count = 0;
    const char *const *dev_ext_names = NULL;
    const void *dev_features = NULL;
    float queue_prio = 1.0f;
    VkDeviceQueueCreateInfo queue_info = {0};
    VkDeviceCreateInfo dev_info = {0};
    VkCommandPoolCreateInfo pool_info = {0};
    EmberwaveVulkanDevice ew_dev = {.struct_size = sizeof ew_dev};
    EmberwaveContext *context = NULL;
    EmberwaveContextInfo ctx_info = {.struct_size = sizeof ctx_info};
    EmberwaveDecoder *decoder = NULL;
    EmberwaveFrameInfo frame_info = {.struct_size = sizeof frame_info};
    VkFormatProperties r16_fmt_props = {0};
    int supports_r16 = 0;
    uint32_t i;
    int exit_code = 1;

    if (argc != 2) {
        fprintf(stderr, "usage: %s INPUT.pwv1\n", argv[0]);
        return 2;
    }

    file_bytes = read_file(argv[1], &file_size);
    if (!file_bytes) {
        return 1;
    }
    status = emberwave_probe_pwv1(file_bytes, file_size, &stream);
    if (status != EMBERWAVE_STATUS_OK) {
        free(file_bytes);
        return fail_ew("emberwave_probe_pwv1", status);
    }
    if (file_size < 28) {
        fprintf(stderr, "%s: missing first frame\n", argv[1]);
        free(file_bytes);
        return 1;
    }
    frame_size = read_le32(file_bytes + 24);
    if (frame_size > file_size - 28) {
        fprintf(stderr, "%s: truncated first frame\n", argv[1]);
        free(file_bytes);
        return 1;
    }
    frame_bytes = file_bytes + 28;

    status = emberwave_vulkan_instance_extensions(
        (EmberwaveVkGetInstanceProcAddr)vkGetInstanceProcAddr, NULL, 0, &inst_ext_count);
    if (status != EMBERWAVE_STATUS_OK) {
        free(file_bytes);
        return fail_ew("emberwave_vulkan_instance_extensions(count)", status);
    }
    if (inst_ext_count > 0) {
        inst_ext_names = calloc(inst_ext_count, sizeof(char *));
        if (!inst_ext_names) {
            free(file_bytes);
            return 1;
        }
        status = emberwave_vulkan_instance_extensions(
            (EmberwaveVkGetInstanceProcAddr)vkGetInstanceProcAddr, inst_ext_names, inst_ext_count,
            &inst_ext_count);
        if (status != EMBERWAVE_STATUS_OK) {
            free(inst_ext_names);
            free(file_bytes);
            return fail_ew("emberwave_vulkan_instance_extensions(fill)", status);
        }
    }

    enum_inst_ver = (PFN_vkEnumerateInstanceVersion)vkGetInstanceProcAddr(
        VK_NULL_HANDLE, "vkEnumerateInstanceVersion");
    if (enum_inst_ver) {
        uint32_t loader_ver = 0;
        if (enum_inst_ver(&loader_ver) == VK_SUCCESS) {
            api_version = loader_ver < VK_API_VERSION_1_3 ? loader_ver : VK_API_VERSION_1_3;
        }
    }

    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "emberwave-android-vulkan-r16";
    app_info.apiVersion = api_version;

    inst_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    inst_info.pApplicationInfo = &app_info;
    inst_info.enabledExtensionCount = inst_ext_count;
    inst_info.ppEnabledExtensionNames = inst_ext_names;

    vk_res = vkCreateInstance(&inst_info, NULL, &vk.instance);
    if (vk_res != VK_SUCCESS) {
        free(inst_ext_names);
        free(file_bytes);
        return fail_vk("vkCreateInstance", vk_res);
    }

    vk_res = vkEnumeratePhysicalDevices(vk.instance, &phys_count, NULL);
    if (vk_res != VK_SUCCESS || phys_count == 0) {
        fprintf(stderr, "no Vulkan physical devices found (%d)\n", (int)vk_res);
        goto cleanup;
    }
    phys_devices = calloc(phys_count, sizeof(VkPhysicalDevice));
    if (!phys_devices) {
        goto cleanup;
    }
    vk_res = vkEnumeratePhysicalDevices(vk.instance, &phys_count, phys_devices);
    if (vk_res != VK_SUCCESS) {
        fail_vk("vkEnumeratePhysicalDevices", vk_res);
        goto cleanup;
    }

    ew_inst.api_version = api_version;
    ew_inst.extension_count = inst_ext_count;
    ew_inst.get_instance_proc_addr = (EmberwaveVkGetInstanceProcAddr)vkGetInstanceProcAddr;
    ew_inst.instance = vk.instance;
    ew_inst.extensions = inst_ext_names;

    for (i = 0; i < phys_count; i++) {
        uint32_t qf_count = 0;
        VkQueueFamilyProperties *qfs = NULL;
        uint32_t q;

        vkGetPhysicalDeviceQueueFamilyProperties(phys_devices[i], &qf_count, NULL);
        if (qf_count == 0) {
            continue;
        }
        qfs = calloc(qf_count, sizeof(VkQueueFamilyProperties));
        if (!qfs) {
            goto cleanup;
        }
        vkGetPhysicalDeviceQueueFamilyProperties(phys_devices[i], &qf_count, qfs);
        for (q = 0; q < qf_count; q++) {
            if ((qfs[q].queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) ==
                (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) {
                EmberwaveVulkanRequirements *candidate = NULL;
                status = emberwave_vulkan_requirements_create(&ew_inst, phys_devices[i], &candidate);
                if (status == EMBERWAVE_STATUS_OK) {
                    vk.physical_device = phys_devices[i];
                    vk.queue_family = q;
                    reqs = candidate;
                    break;
                }
            }
        }
        free(qfs);
        if (reqs != NULL) {
            break;
        }
    }

    if (reqs == NULL) {
        fprintf(stderr, "no physical device accepted by emberwave_vulkan_requirements_create: %s\n",
                emberwave_last_error());
        goto cleanup;
    }

    vkGetPhysicalDeviceProperties(vk.physical_device, &phys_props);
    vkGetPhysicalDeviceMemoryProperties(vk.physical_device, &vk.mem_props);
    vkGetPhysicalDeviceFormatProperties(vk.physical_device, VK_FORMAT_R16_UNORM, &r16_fmt_props);

    printf("Vulkan device: %s (vendor 0x%04x, device 0x%04x, api %u.%u.%u, driver 0x%x)\n",
           phys_props.deviceName, phys_props.vendorID, phys_props.deviceID,
           VK_VERSION_MAJOR(phys_props.apiVersion), VK_VERSION_MINOR(phys_props.apiVersion),
           VK_VERSION_PATCH(phys_props.apiVersion), phys_props.driverVersion);
    printf("VK_FORMAT_R16_UNORM optimalTilingFeatures: 0x%x "
           "(SAMPLED=%d STORAGE=%d TRANSFER_SRC=%d TRANSFER_DST=%d)\n",
           (unsigned)r16_fmt_props.optimalTilingFeatures,
           (r16_fmt_props.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) != 0,
           (r16_fmt_props.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) != 0,
           (r16_fmt_props.optimalTilingFeatures & VK_FORMAT_FEATURE_TRANSFER_SRC_BIT) != 0,
           (r16_fmt_props.optimalTilingFeatures & VK_FORMAT_FEATURE_TRANSFER_DST_BIT) != 0);

    dev_ext_names = emberwave_vulkan_requirements_extensions(reqs, &dev_ext_count);
    dev_features = emberwave_vulkan_requirements_features(reqs);

    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.queueFamilyIndex = vk.queue_family;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &queue_prio;

    dev_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dev_info.pNext = dev_features;
    dev_info.queueCreateInfoCount = 1;
    dev_info.pQueueCreateInfos = &queue_info;
    dev_info.enabledExtensionCount = dev_ext_count;
    dev_info.ppEnabledExtensionNames = dev_ext_names;

    vk_res = vkCreateDevice(vk.physical_device, &dev_info, NULL, &vk.device);
    if (vk_res != VK_SUCCESS) {
        fail_vk("vkCreateDevice", vk_res);
        goto cleanup;
    }
    vkGetDeviceQueue(vk.device, vk.queue_family, 0, &vk.queue);

    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_info.queueFamilyIndex = vk.queue_family;
    vk_res = vkCreateCommandPool(vk.device, &pool_info, NULL, &vk.command_pool);
    if (vk_res != VK_SUCCESS) {
        fail_vk("vkCreateCommandPool", vk_res);
        goto cleanup;
    }

    ew_dev.instance = &ew_inst;
    ew_dev.physical_device = vk.physical_device;
    ew_dev.device = vk.device;
    ew_dev.extensions = dev_ext_names;
    ew_dev.extension_count = dev_ext_count;
    ew_dev.queue_family_index = vk.queue_family;
    ew_dev.queue_index = 0;

    status = emberwave_context_create_vulkan(&ew_dev, &context);
    emberwave_vulkan_requirements_destroy(reqs);
    reqs = NULL;
    if (status != EMBERWAVE_STATUS_OK) {
        fail_ew("emberwave_context_create_vulkan", status);
        goto cleanup;
    }

    status = emberwave_context_info(context, &ctx_info);
    if (status != EMBERWAVE_STATUS_OK) {
        fail_ew("emberwave_context_info", status);
        goto cleanup;
    }
    supports_r16 = (ctx_info.features & EMBERWAVE_CONTEXT_FEATURE_PLANE_R16) != 0;
    printf("emberwave %s adapter: %s (backend=%u, kind=%u, features=0x%x, PLANE_R16=%d)\n",
           emberwave_version(), emberwave_context_adapter(context), ctx_info.backend,
           ctx_info.kind, ctx_info.features, supports_r16);

    status = emberwave_decoder_create(context, stream.width, stream.height, stream.chroma, &decoder);
    if (status != EMBERWAVE_STATUS_OK) {
        fail_ew("emberwave_decoder_create", status);
        goto cleanup;
    }
    status = emberwave_decoder_info(decoder, &frame_info);
    if (status != EMBERWAVE_STATUS_OK) {
        fail_ew("emberwave_decoder_info", status);
        goto cleanup;
    }
    printf("stream %ux%u chroma=%u (chroma plane %ux%u)\n", frame_info.width, frame_info.height,
           frame_info.chroma, frame_info.chroma_width, frame_info.chroma_height);

    status = emberwave_decoder_push_packet(decoder, frame_bytes, frame_size);
    if (status != EMBERWAVE_STATUS_OK) {
        fail_ew("emberwave_decoder_push_packet", status);
        goto cleanup;
    }
    if (!emberwave_decoder_is_ready(decoder, false)) {
        fprintf(stderr, "decoder not ready after pushing first frame\n");
        goto cleanup;
    }

    if (!supports_r16) {
        EmberwaveTexture *probe_tex = NULL;
        PlaneImage r16_probe = {0};
        PlaneImage r8_imgs[3] = {{0}};
        EmberwaveTexture *r8_tex[3] = {NULL, NULL, NULL};
        uint32_t widths[3];
        uint32_t heights[3];
        uint32_t planes[3] = {EMBERWAVE_PLANE_Y, EMBERWAVE_PLANE_CB, EMBERWAVE_PLANE_CR};

        status = emberwave_texture_create_plane_with_format(
            decoder, EMBERWAVE_PLANE_Y, EMBERWAVE_PLANE_FORMAT_R16, &probe_tex);
        if (status != EMBERWAVE_STATUS_UNSUPPORTED || probe_tex != NULL) {
            fprintf(stderr, "expected EMBERWAVE_STATUS_UNSUPPORTED for owned R16, got %d\n",
                    (int)status);
            if (probe_tex != NULL) {
                emberwave_texture_destroy(probe_tex);
            }
            goto cleanup;
        }
        printf("owned R16 plane creation cleanly returned EMBERWAVE_STATUS_UNSUPPORTED (%s)\n",
               emberwave_last_error());

        if ((r16_fmt_props.optimalTilingFeatures &
             (VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT)) ==
            (VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT)) {
            if (create_plane_image(&vk, VK_FORMAT_R16_UNORM,
                                   VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                                       VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                                   frame_info.width, frame_info.height, 2, &r16_probe) == 0) {
                EmberwaveVulkanImage desc = {.struct_size = sizeof desc};
                desc.image = (uint64_t)(uintptr_t)r16_probe.image;
                desc.format = VK_FORMAT_R16_UNORM;
                desc.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                             VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
                desc.layout = VK_IMAGE_LAYOUT_UNDEFINED;
                status = emberwave_texture_from_vulkan_plane(decoder, EMBERWAVE_PLANE_Y, &desc,
                                                             &probe_tex);
                destroy_plane_image(&vk, &r16_probe);
                if (status != EMBERWAVE_STATUS_UNSUPPORTED || probe_tex != NULL) {
                    fprintf(stderr,
                            "expected EMBERWAVE_STATUS_UNSUPPORTED for wrapped R16, got %d\n",
                            (int)status);
                    if (probe_tex != NULL) {
                        emberwave_texture_destroy(probe_tex);
                    }
                    goto cleanup;
                }
                printf("wrapped R16 plane creation cleanly returned "
                       "EMBERWAVE_STATUS_UNSUPPORTED\n");
            }
        }

        /* Fallback verification: decode into wrapped R8_UNORM Vulkan images. */
        widths[0] = frame_info.width;
        heights[0] = frame_info.height;
        widths[1] = widths[2] = frame_info.chroma_width;
        heights[1] = heights[2] = frame_info.chroma_height;
        for (i = 0; i < 3; i++) {
            EmberwaveVulkanImage desc = {.struct_size = sizeof desc};
            VkImageUsageFlags usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                      VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
            if (create_plane_image(&vk, VK_FORMAT_R8_UNORM, usage, widths[i], heights[i], 1,
                                   &r8_imgs[i]) != 0) {
                for (uint32_t j = 0; j < 3; j++) {
                    emberwave_texture_destroy(r8_tex[j]);
                    destroy_plane_image(&vk, &r8_imgs[j]);
                }
                goto cleanup;
            }
            desc.image = (uint64_t)(uintptr_t)r8_imgs[i].image;
            desc.format = VK_FORMAT_R8_UNORM;
            desc.usage = usage;
            desc.layout = VK_IMAGE_LAYOUT_UNDEFINED;
            desc.release = on_image_release;
            status = emberwave_texture_from_vulkan_plane(decoder, planes[i], &desc, &r8_tex[i]);
            if (status != EMBERWAVE_STATUS_OK) {
                fail_ew("emberwave_texture_from_vulkan_plane(R8)", status);
                for (uint32_t j = 0; j < 3; j++) {
                    emberwave_texture_destroy(r8_tex[j]);
                    destroy_plane_image(&vk, &r8_imgs[j]);
                }
                goto cleanup;
            }
        }
        status = emberwave_decoder_decode_plane_textures(decoder, r8_tex[0], r8_tex[1], r8_tex[2]);
        if (status != EMBERWAVE_STATUS_OK) {
            fail_ew("emberwave_decoder_decode_plane_textures(R8)", status);
            for (i = 0; i < 3; i++) {
                emberwave_texture_destroy(r8_tex[i]);
                destroy_plane_image(&vk, &r8_imgs[i]);
            }
            goto cleanup;
        }
        {
            size_t luma_px = (size_t)widths[0] * heights[0];
            size_t chroma_px = (size_t)widths[1] * heights[1];
            unsigned char *cpu_yuv = malloc(luma_px + 2 * chroma_px);
            EmberwavePlane y_cpu, cb_cpu, cr_cpu;
            if (!cpu_yuv) {
                for (i = 0; i < 3; i++) {
                    emberwave_texture_destroy(r8_tex[i]);
                    destroy_plane_image(&vk, &r8_imgs[i]);
                }
                goto cleanup;
            }
            y_cpu.data = cpu_yuv;
            y_cpu.stride = widths[0];
            y_cpu.size = luma_px;
            cb_cpu.data = cpu_yuv + luma_px;
            cb_cpu.stride = widths[1];
            cb_cpu.size = chroma_px;
            cr_cpu.data = cpu_yuv + luma_px + chroma_px;
            cr_cpu.stride = widths[2];
            cr_cpu.size = chroma_px;
            status = emberwave_decoder_decode_yuv(decoder, y_cpu, cb_cpu, cr_cpu);
            if (status != EMBERWAVE_STATUS_OK) {
                fail_ew("emberwave_decoder_decode_yuv(R8 fallback)", status);
                free(cpu_yuv);
                for (i = 0; i < 3; i++) {
                    emberwave_texture_destroy(r8_tex[i]);
                    destroy_plane_image(&vk, &r8_imgs[i]);
                }
                goto cleanup;
            }
            for (i = 0; i < 3; i++) {
                size_t count = (size_t)widths[i] * heights[i];
                unsigned char *gpu_u8 = malloc(count);
                const unsigned char *ref_u8 =
                    (i == 0) ? y_cpu.data : ((i == 1) ? cb_cpu.data : cr_cpu.data);
                if (!gpu_u8 ||
                    readback_image(&vk, r8_imgs[i].image, widths[i], heights[i], 1, gpu_u8) != 0 ||
                    memcmp(gpu_u8, ref_u8, count) != 0) {
                    fprintf(stderr, "plane %u: R8 fallback readback mismatch\n", i);
                    free(gpu_u8);
                    free(cpu_yuv);
                    for (uint32_t j = 0; j < 3; j++) {
                        emberwave_texture_destroy(r8_tex[j]);
                        destroy_plane_image(&vk, &r8_imgs[j]);
                    }
                    goto cleanup;
                }
                free(gpu_u8);
            }
            free(cpu_yuv);
        }
        for (i = 0; i < 3; i++) {
            emberwave_texture_destroy(r8_tex[i]);
        }
        printf("R8 fallback decode + readback succeeded (PLANE_R16 unsupported on this device)\n");
        emberwave_decoder_destroy(decoder);
        decoder = NULL;
        emberwave_context_destroy(context);
        context = NULL;
        for (i = 0; i < 3; i++) {
            destroy_plane_image(&vk, &r8_imgs[i]);
        }
        if (g_released_count != 3) {
            fprintf(stderr, "expected 3 image release callbacks, got %u\n", g_released_count);
            goto cleanup;
        }
        exit_code = 0;
        goto cleanup;
    }

    {
        PlaneImage r16_imgs[3] = {{0}};
        EmberwaveTexture *wrapped[3] = {NULL, NULL, NULL};
        EmberwaveTexture *owned[3] = {NULL, NULL, NULL};
        uint32_t widths[3] = {frame_info.width, frame_info.chroma_width, frame_info.chroma_width};
        uint32_t heights[3] = {frame_info.height, frame_info.chroma_height,
                               frame_info.chroma_height};
        uint32_t planes[3] = {EMBERWAVE_PLANE_Y, EMBERWAVE_PLANE_CB, EMBERWAVE_PLANE_CR};
        VkImageUsageFlags r16_usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                                      VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        unsigned char *cpu_yuv = NULL;
        size_t luma_px = (size_t)widths[0] * heights[0];
        size_t chroma_px = (size_t)widths[1] * heights[1];
        EmberwavePlane y_cpu, cb_cpu, cr_cpu;

        for (i = 0; i < 3; i++) {
            EmberwaveVulkanImage desc = {.struct_size = sizeof desc};
            if (create_plane_image(&vk, VK_FORMAT_R16_UNORM, r16_usage, widths[i], heights[i], 2,
                                   &r16_imgs[i]) != 0) {
                goto r16_cleanup;
            }
            desc.image = (uint64_t)(uintptr_t)r16_imgs[i].image;
            desc.format = VK_FORMAT_R16_UNORM;
            desc.usage = r16_usage;
            desc.layout = VK_IMAGE_LAYOUT_UNDEFINED;
            desc.release = on_image_release;
            status = emberwave_texture_from_vulkan_plane(decoder, planes[i], &desc, &wrapped[i]);
            if (status != EMBERWAVE_STATUS_OK) {
                fail_ew("emberwave_texture_from_vulkan_plane(R16)", status);
                goto r16_cleanup;
            }
            status = emberwave_texture_create_plane_with_format(
                decoder, planes[i], EMBERWAVE_PLANE_FORMAT_R16, &owned[i]);
            if (status != EMBERWAVE_STATUS_OK) {
                fail_ew("emberwave_texture_create_plane_with_format(R16)", status);
                goto r16_cleanup;
            }
        }

        status = emberwave_decoder_decode_plane_textures(decoder, wrapped[0], wrapped[1],
                                                         wrapped[2]);
        if (status != EMBERWAVE_STATUS_OK) {
            fail_ew("emberwave_decoder_decode_plane_textures(wrapped R16)", status);
            goto r16_cleanup;
        }
        printf("emberwave_decoder_decode_plane_textures(wrapped R16): OK\n");

        status = emberwave_decoder_decode_plane_textures(decoder, owned[0], owned[1], owned[2]);
        if (status != EMBERWAVE_STATUS_OK) {
            fail_ew("emberwave_decoder_decode_plane_textures(owned R16)", status);
            goto r16_cleanup;
        }
        printf("emberwave_decoder_decode_plane_textures(owned R16): OK\n");

        cpu_yuv = malloc(luma_px + 2 * chroma_px);
        if (!cpu_yuv) {
            goto r16_cleanup;
        }
        y_cpu.data = cpu_yuv;
        y_cpu.stride = widths[0];
        y_cpu.size = luma_px;
        cb_cpu.data = cpu_yuv + luma_px;
        cb_cpu.stride = widths[1];
        cb_cpu.size = chroma_px;
        cr_cpu.data = cpu_yuv + luma_px + chroma_px;
        cr_cpu.stride = widths[2];
        cr_cpu.size = chroma_px;
        status = emberwave_decoder_decode_yuv(decoder, y_cpu, cb_cpu, cr_cpu);
        if (status != EMBERWAVE_STATUS_OK) {
            fail_ew("emberwave_decoder_decode_yuv", status);
            goto r16_cleanup;
        }

        for (i = 0; i < 3; i++) {
            size_t count = (size_t)widths[i] * heights[i];
            size_t bytes = count * 2;
            uint16_t *wrapped_u16 = malloc(bytes);
            uint16_t *owned_u16 = malloc(bytes);
            uint64_t owned_raw = emberwave_texture_vulkan(owned[i]);
            const unsigned char *ref_u8 =
                (i == 0) ? y_cpu.data : ((i == 1) ? cb_cpu.data : cr_cpu.data);
            uint16_t min_v = UINT16_MAX;
            uint16_t max_v = 0;
            uint32_t max_u8_diff = 0;
            size_t non_u8_grid = 0;
            size_t p;

            if (!wrapped_u16 || !owned_u16 || owned_raw == 0) {
                free(wrapped_u16);
                free(owned_u16);
                goto r16_cleanup;
            }
            if (readback_image(&vk, r16_imgs[i].image, widths[i], heights[i], 2,
                               (unsigned char *)wrapped_u16) != 0 ||
                readback_image(&vk, (VkImage)(uintptr_t)owned_raw, widths[i], heights[i], 2,
                               (unsigned char *)owned_u16) != 0) {
                free(wrapped_u16);
                free(owned_u16);
                goto r16_cleanup;
            }
            if (memcmp(wrapped_u16, owned_u16, bytes) != 0) {
                fprintf(stderr, "plane %u: wrapped and owned R16 readbacks differ\n", i);
                free(wrapped_u16);
                free(owned_u16);
                goto r16_cleanup;
            }

            for (p = 0; p < count; p++) {
                uint16_t v = wrapped_u16[p];
                uint32_t q8 = ((uint32_t)v * 255u + 32767u) / 65535u;
                uint32_t ref = ref_u8[p];
                uint32_t diff = q8 > ref ? q8 - ref : ref - q8;
                if (v < min_v) {
                    min_v = v;
                }
                if (v > max_v) {
                    max_v = v;
                }
                if (diff > max_u8_diff) {
                    max_u8_diff = diff;
                }
                if (v % 257u != 0) {
                    non_u8_grid++;
                }
            }
            printf("plane %u (%ux%u R16): min=%u max=%u max_u8_diff=%u non_u8_grid=%zu/%zu "
                   "first=[%u,%u,%u,%u]\n",
                   i, widths[i], heights[i], (unsigned)min_v, (unsigned)max_v, max_u8_diff,
                   non_u8_grid, count, (unsigned)wrapped_u16[0], (unsigned)wrapped_u16[1],
                   (unsigned)wrapped_u16[2], (unsigned)wrapped_u16[3]);
            free(wrapped_u16);
            free(owned_u16);
            if (max_u8_diff > 1) {
                fprintf(stderr, "plane %u: R16 vs 8-bit YUV diff %u > 1\n", i, max_u8_diff);
                goto r16_cleanup;
            }
        }

        exit_code = 0;
    r16_cleanup:
        free(cpu_yuv);
        for (i = 0; i < 3; i++) {
            emberwave_texture_destroy(wrapped[i]);
            emberwave_texture_destroy(owned[i]);
        }
        emberwave_decoder_destroy(decoder);
        decoder = NULL;
        emberwave_context_destroy(context);
        context = NULL;
        for (i = 0; i < 3; i++) {
            destroy_plane_image(&vk, &r16_imgs[i]);
        }
        if (exit_code == 0 && g_released_count != 3) {
            fprintf(stderr, "expected 3 image release callbacks, got %u\n", g_released_count);
            exit_code = 1;
        }
    }

cleanup:
    if (reqs != NULL) {
        emberwave_vulkan_requirements_destroy(reqs);
    }
    if (decoder != NULL) {
        emberwave_decoder_destroy(decoder);
    }
    if (context != NULL) {
        emberwave_context_destroy(context);
    }
    if (vk.command_pool != VK_NULL_HANDLE) {
        vkDestroyCommandPool(vk.device, vk.command_pool, NULL);
    }
    if (vk.device != VK_NULL_HANDLE) {
        vkDestroyDevice(vk.device, NULL);
    }
    if (vk.instance != VK_NULL_HANDLE) {
        vkDestroyInstance(vk.instance, NULL);
    }
    free(phys_devices);
    free((void *)inst_ext_names);
    free(file_bytes);
    return exit_code;
}
