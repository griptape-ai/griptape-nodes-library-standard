---
description: Reads a text file from disk and outputs its contents as text.
when_to_use: Use it when you want to bring an existing file, such as an EDL, notes or a CSV, into your workflow as text.
---

## When to use it

You have a text file and want its content in your workflow, for example an EDL or a shot list to hand to an [Agent](../agents/Agent.md). PDFs also work, and their pages are joined into one text. To get more detail from a PDF, such as the page count, use [Load PDF](../files/LoadPdf.md).

## Example

1. Add **Load Text** and set `path` to a shot list file, such as `shot_list.csv`.
1. Connect `output` to an Agent and ask it to summarize which shots are still in progress.

## Parameters

### path

The file to read. The file picker offers `.txt`, `.edl`, `.md`, `.csv`, `.json`, `.yaml`, `.yml`, `.htm`, `.html`, `.xml`, `.env`, `.py` and `.pdf`. Files inside the project are stored as project-relative paths. A relative path is read from the workspace directory.

### output

The text of the file. For a PDF, the text of each page, separated by a blank line. The output is also written to the node's `path` value after the run, as the resolved path.
