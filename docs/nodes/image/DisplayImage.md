## What is it?

The DisplayImage node displays an image and provides its dimensions. It's a simple viewer that shows an image in your workflow while also extracting useful information like width and height.

## When would I use it?

Use this node when you want to:

- Display an image in your workflow for visual feedback
- Get the dimensions (width and height) of an image
- Pass an image through your workflow while viewing it
- Debug your workflow by seeing what image is being processed

## How to use it

### Basic Setup

1. Add the DisplayImage node to your workflow
1. Connect an image source to the "image" input
1. The image will be displayed and its dimensions will be available as outputs (note: They are hidden by default. You can display them in the Properties panel on the right.)

### Parameters

- **image**: The image to display

### Outputs

- **image**: The same image that was input, passed through for use by other nodes
- **width**: The width of the image in pixels (integer)
- **height**: The height of the image in pixels (integer)

## Example

A common workflow pattern:

1. Connect an image source (like GenerateImage or LoadImage) to the DisplayImage node
1. View the image in the DisplayImage node to confirm it's what you expected
1. Use the width and height outputs if you need the image dimensions for other nodes
1. Connect the image output to other nodes that need the image

## Important Notes

- The DisplayImage node passes the image through unchanged - it's purely for display and dimension extraction
- For ImageUrlArtifact inputs, the node reads the file at the artifact's location to determine dimensions
- For ImageArtifact inputs, dimensions are read directly from the artifact's properties
- The width and height outputs are hidden from the UI by default, but can be enabled in the Properties Panel.

## Common Issues

- **Image not displaying**: Check that your image source is properly connected and producing valid image data
- **Dimensions showing as 0**: No image is connected, the file could not be read, or the image is an SVG. The node cannot measure SVG files and reports 0 for both
- **Slow performance with URLs**: The node reads the whole image to measure it, so a large remote image can take time to load
