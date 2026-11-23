---
description: Returns the path of your system's temporary folder.
when_to_use: Use it when a workflow needs a scratch location that works on Windows, macOS and Linux.
---

## When to use it

You need somewhere to write files you don't plan to keep. This node gives you the operating system's temp folder, such as `/tmp` or `C:\Users\you\AppData\Local\Temp`, so you don't have to hard-code it. Pair it with [PathJoin](PathJoin.md) or [TempFilename](TempFilename.md) to build a full path.

## Parameters

### path

The temp folder path. The node reads it from the system each time it runs. It does not create anything.
