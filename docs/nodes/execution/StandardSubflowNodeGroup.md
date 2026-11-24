---
description: Groups nodes into one node that runs everything inside it in parallel.
when_to_use: Use it when you want to tidy part of a workflow into one node, or run independent steps at the same time.
---

## When to use it

Select nodes and group them to collapse part of a workflow into a single node. Connections that cross the group boundary become parameters on the group. When the group runs, the nodes inside run together as a subflow and their outputs are passed back out.

## Parameters

### execution_environment

Where the group runs. The choices depend on the libraries and publishers installed, and the default is local execution.
