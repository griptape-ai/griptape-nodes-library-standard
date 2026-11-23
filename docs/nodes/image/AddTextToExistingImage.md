## What is it?

The AddTextToExistingImage node overlays text onto an existing image. It supports text/background colors (with alpha), alignment controls, and simple template placeholders like `{key}` that are expanded from a separate dictionary input.

## When would I use it?

Use this node when you want to:

- Add titles, captions, or labels onto an image
- Stamp metadata (like filenames, timestamps, or IDs) onto images
- Create annotated image outputs for downstream steps
- Render dynamic text using `{key}` placeholders sourced from a dictionary

## How to use it

### Basic Setup

1. Add the AddTextToExistingImage node to your workflow
1. Connect an image source to **input_image**
1. Enter the text template in **text**
1. (Optional) Connect a dictionary to **template_values** to expand `{key}` placeholders
1. Adjust alignment, colors, border, and font size
1. Run the node to produce **output**

### Parameters

#### Required Inputs

- **input_image**: The image to render text onto (ImageUrlArtifact / ImageArtifact / dict)

#### Text Inputs

- **text** (string): The text template to render

    - Supports placeholders like `{key_name}`
    - Placeholder expansion is applied only to what’s rendered on the image; the `text` output parameter remains the original template

- **template_values** (dict, optional): Values used to expand placeholders in `text`

    - If a placeholder key is not present, it will be left as-is in the rendered text
    - Missing keys are also reported in `result_details` when the node runs

#### Styling Parameters

- **text_color** (hexa, default: `#ffffffff`): Text color including alpha
- **text_background** (hexa, default: `#000000ff`): Background rectangle color behind the text including alpha
- **text_vertical_alignment** (top | center | bottom, default: top): Vertical alignment of the text block
- **text_horizontal_alignment** (left | center | right, default: left): Horizontal alignment of the text block
- **margin** (int, default: 10): Margin inset from image edges for text placement
- **font_size** (int, default: 36): Font size used for rendering

### Outputs

- **output**: The updated image (ImageUrlArtifact)
- **text**: The original text template string (not the expanded render string)
- **was_successful**: Indicates whether the node succeeded
- **result_details**: Success/failure details, including missing placeholder key messages

## Example

A typical burn-in workflow for a dailies frame:

1. Load a rendered frame using LoadImage

1. Add AddTextToExistingImage

1. Set **text** to:

    `"{shot}  |  v{version}  |  {status}"`

1. Provide **template_values**:

    ```
    {"shot": "SH010", "version": "003", "status": "Pending review"}
    ```

1. Set **text_vertical_alignment** to `bottom`

1. Set `text_background` to a semi-transparent black like `#00000080`

1. Set `text_color` to white `#ffffffff`

1. Run the node and connect **output** to DisplayImage

## Important Notes

- **Missing keys**: If `{key}` is not present in `template_values`, the placeholder stays in the rendered text and a message like `key: key not found in dictionary input` is appended to `result_details`.
- **Alpha support**: Both text and background colors support transparency via hexa (`#RRGGBBAA`).
- **Output file**: The result is always a PNG, written to the location set by `output_file`.
- **Placeholders**: A placeholder is `{` plus letters, digits or underscores plus `}`. Keys with other characters, such as spaces or dashes, are not expanded.
- **Font**: The node uses Pillow's built-in font. `font_size` is the only font setting.
- **No text**: If `text` is empty after expansion, the output is the input image saved as a PNG.
- **Validation**: `margin` must be 0 or more and `font_size` must be greater than 0, otherwise the node fails.
- **Preview**: The node shows a live preview of the text as you change the settings.
