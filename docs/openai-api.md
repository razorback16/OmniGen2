# Image Generation API Guide

## Generate Images

You can use the image generation endpoint to create images based on text prompts. To learn more about customizing the output (size, quality, format, transparency), refer to the [Customize Image Output](#customize-image-output) section below.

- You can set the `n` parameter to generate multiple images at once in a single request (by default, the API returns a single image).

**Example: Generate an Image**
```bash
curl -X POST "https://api.openai.com/v1/images/generations" \
    -H "Authorization: Bearer $OPENAI_API_KEY" \
    -H "Content-type: application/json" \
    -d '{
        "model": "gpt-image-1",
        "prompt": "A childrens book drawing of a veterinarian using a stethoscope to listen to the heartbeat of a baby otter."
    }' | jq -r '.data[0].b64_json' | base64 --decode > otter.png
```

---

## Edit Images

The image edits endpoint lets you:

- Edit existing images
- Generate new images using other images as a reference
- Edit parts of an image by uploading an image and mask indicating which areas should be replaced (a process known as **inpainting**)
- Create a new image using image references

You can use one or more images as a reference to generate a new image.

**Example: Combine Multiple Images**
```bash
curl -s -D >(grep -i x-request-id >&2) \
  -o >(jq -r '.data[0].b64_json' | base64 --decode > gift-basket.png) \
  -X POST "https://api.openai.com/v1/images/edits" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -F "model=gpt-image-1" \
  -F "image[]=@body-lotion.png" \
  -F "image[]=@bath-bomb.png" \
  -F "image[]=@incense-kit.png" \
  -F "image[]=@soap.png" \
  -F 'prompt=Generate a photorealistic image of a gift basket on a white background labeled "Relax & Unwind" with a ribbon and handwriting-like font, containing all the items in the reference pictures'
```

### Edit an Image Using a Mask (Inpainting)

You can provide a mask to indicate where the image should be edited. The transparent areas of the mask will be replaced, while the filled areas will be left unchanged.

- The prompt describes what you want the final edited image to be or what you want to edit specifically.
- If you provide multiple input images, the mask will be applied to the first image.

**Example: Inpainting**
```bash
curl -s -D >(grep -i x-request-id >&2) \
  -o >(jq -r '.data[0].b64_json' | base64 --decode > lounge.png) \
  -X POST "https://api.openai.com/v1/images/edits" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -F "model=gpt-image-1" \
  -F "mask=@mask.png" \
  -F "image[]=@sunlit_lounge.png" \
  -F 'prompt=A sunlit indoor lounge area with a pool containing a flamingo'
```

---

## Mask Requirements

- The image to edit and mask must be of the same format and size (less than 25MB).
- The mask image must contain an alpha channel. If you're using an image editing tool to create the mask, make sure to save the mask with an alpha channel.

### Add an Alpha Channel to a Black and White Mask

You can modify a black and white image programmatically to add an alpha channel:

```python
from PIL import Image
from io import BytesIO

# 1. Load your black & white mask as a grayscale image
mask = Image.open(img_path_mask).convert("L")

# 2. Convert it to RGBA so it has space for an alpha channel
mask_rgba = mask.convert("RGBA")

# 3. Then use the mask itself to fill that alpha channel
mask_rgba.putalpha(mask)

# 4. Convert the mask into bytes
buf = BytesIO()
mask_rgba.save(buf, format="PNG")
mask_bytes = buf.getvalue()

# 5. Save the resulting file
img_path_mask_alpha = "mask_alpha.png"
with open(img_path_mask_alpha, "wb") as f:
    f.write(mask_bytes)
```

---

## Customize Image Output

You can configure the following output options:

- **Size:** Image dimensions (e.g., 1024x1024, 1024x1536)
- **Quality:** Rendering quality (e.g. low, medium, high)
- **Format:** File output format
- **Compression:** Compression level (0-100%) for JPEG and WebP formats
- **Background:** Transparent or opaque

`size`, `quality`, and `background` support the `auto` option, where the model will automatically select the best option based on the prompt.

### Size and Quality Options

- Square images with standard quality are the fastest to generate.
- The default size is **1024x1024** pixels.

**Available Sizes:**
- 1024x1024 (square)
- 1536x1024 (landscape)
- 1024x1536 (portrait)
- auto (default)

**Quality Options:**
- low
- medium
- high
- auto (default)

### Output Format

The Image API returns base64-encoded image data. The default format is **png**, but you can also request **jpeg** or **webp**.

- If using jpeg or webp, you can also specify the `output_compression` parameter to control the compression level (0-100%).  
  For example, `output_compression=50` will compress the image by 50%.

### Transparency

The `gpt-image-1` model supports transparent backgrounds. To enable transparency, set the `background` parameter to `transparent`.

- Supported only with the **png** and **webp** output formats.
- Works best when setting the quality to **medium** or **high**.

**Example: Transparent Background**
```bash
curl -X POST "https://api.openai.com/v1/images" \
    -H "Authorization: Bearer $OPENAI_API_KEY" \
    -H "Content-type: application/json" \
    -d '{
        "prompt": "Draw a 2D pixel art style sprite sheet of a tabby gray cat",
        "quality": "high",
        "size": "1024x1024",
        "background": "transparent"
    }' | jq -r 'data[0].b64_json' | base64 --decode > sprite.png
```

---

## Limitations

The GPT-4o Image model is a powerful and versatile image generation model, but it still has some limitations to be aware of:

- **Latency:** Complex prompts may take up to 2 minutes to process.
- **Text Rendering:** Although significantly improved over the DALL·E series, the model can still struggle with precise text placement and clarity.
- **Consistency:** While capable of producing consistent imagery, the model may occasionally struggle to maintain visual consistency for recurring characters or brand elements across multiple generations.
- **Composition Control:** Despite improved instruction following, the model may have difficulty placing elements precisely in structured or layout-sensitive compositions.

---

## Content Moderation

All prompts and generated images are filtered in accordance with our content policy.

For image generation using `gpt-image-1`, you can control moderation strictness with the `moderation` parameter. This parameter supports two values:

- `auto` (default): Standard filtering that seeks to limit creating certain categories of potentially age-inappropriate content.
- `low`: Less restrictive filtering.

---

## Cost and Latency

This model generates images by first producing specialized image tokens. Both latency and eventual cost are proportional to the number of tokens required to render an image—larger image sizes and higher quality settings result in more tokens.

The number of tokens generated depends on image dimensions and quality:

| Quality | Square (1024×1024) | Portrait (1024×1536) | Landscape (1536×1024) |
|---------|--------------------|----------------------|-----------------------|
| Low     | 272 tokens         | 408 tokens           | 400 tokens            |
| Medium  | 1056 tokens        | 1584 tokens          | 1568 tokens           |
| High    | 4160 tokens        | 6240 tokens          | 6208 tokens           |

Note that you will also need to account for input tokens: text tokens for the prompt and image tokens for the input images if editing images.

So the final cost is the sum of:
- Input text tokens
- Input image tokens (if using the edits endpoint)
- Image output tokens

**Image Token Pricing (per 1M tokens):**

| Model         | Input (per 1M tokens) | Output (per 1M tokens) |
|---------------|----------------------|------------------------|
| gpt-image-1   | $10.00               | $40.00                 |

Refer to the [OpenAI Pricing page](https://openai.com/pricing) for the most up-to-date rates.

**Latency:** Generation time varies by prompt complexity, image size, and quality. Most requests complete in 10–60 seconds, but complex or high-resolution images may take up to 2 minutes.

For best performance, use standard sizes and quality settings. Batch requests (using the `n` parameter) may increase total latency proportionally to the number of images requested.