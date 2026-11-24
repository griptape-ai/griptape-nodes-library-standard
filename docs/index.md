---
title: "Standard Library"
---

The Standard Library holds the default nodes that ship with Griptape Nodes: loading and showing images, video and audio, generating media with AI models, running Agents, and the data, list, file and flow nodes that connect them.

Every node has a page. It says when to use the node, what it needs to run, and how its parameters behave. A node that calls an AI service says so under **What it needs**, usually a Griptape Nodes license or a Griptape Cloud API key.

## Where to start

- **Ask an AI something inside a workflow.** Use [Agent](nodes/agents/Agent.md). Give it rules with [Ruleset](nodes/rules/Ruleset.md), or make it return structured data with [Create Agent Schema](nodes/agents/create_agent_schema.md).
- **Look at an image.** [Load Image](nodes/image/LoadImage.md), then [Display Image](nodes/image/DisplayImage.md). [Compare Images](nodes/image/CompareImages.md) checks two images against each other.
- **Generate an image.** [Generate Image](nodes/image/GenerateImage.md) makes an image from a prompt with an OpenAI GPT Image model through Griptape Cloud. [OpenAI Image Generation](nodes/image/OpenAiImageGeneration.md) and [Seedream Image Generation](nodes/image/SeedreamImageGeneration.md) are two of the model-specific nodes.
- **Work with video.** [Load Video](nodes/video/LoadVideo.md), [Trim Video](nodes/video/TrimVideo.md), [Resize Video](nodes/video/ResizeVideo.md), then [Save Video](nodes/video/SaveVideo.md).
- **Transcribe a recording or a video's audio.** [Load Audio](nodes/audio/LoadAudio.md) or [Load Video](nodes/video/LoadVideo.md), then [Transcribe Audio](nodes/audio/TranscribeAudio.md).
- **Pull a field out of data.** [JSON Extract Value](nodes/json/JsonExtractValue.md) reads one value or a list of values from JSON. [YAML Extract Value](nodes/yaml/YamlExtractValue.md), [XML Extract Value](nodes/xml/XmlExtractValue.md) and [HTML Extract Value](nodes/html/HtmlExtractValue.md) work the same way.
- **Repeat work for each item.** [ForEach Start](nodes/execution/ForEachStartNode.md) loops over a list, and [IfElse](nodes/execution/IfElse.md) chooses a path.
- **Find or tidy files.** [List Files](nodes/files/ListFiles.md) lists a folder.
- **Leave a note on the canvas.** [Note](nodes/utils/Note.md).

## Categories

The editor groups the nodes by these categories.

| Category | What it holds |
| --- | --- |
| Agent, Memory, Prompt Models, Rules, Tools | Agents and what they run on: memory, model configurations, rulesets and tools. |
| Image, Image Models | Loading, editing, generating and checking images. |
| Video, Seedance | Loading, editing and generating video. |
| Audio | Loading, saving, mixing, transcribing and generating audio. |
| 3D, Splat | Loading 3D models and generating 3D assets and worlds. |
| Text, Classification | Text input, merging, searching, web nodes and yes/no and rating checks. |
| Lists, Dictionary, Convert, Number | Building and changing lists and dictionaries, converting types, and numbers. |
| JSON, YAML, XML, HTML | Reading values in structured data, and replacing them in JSON, YAML and XML. |
| Files, Filesystem, Sequence | Files, folders, output locations and numbered file sequences. |
| Execution Flow, Variables, Workflows, Engine, misc | Loops, branches, variables, subflows, engine control and running Python, notes and grouping. |
