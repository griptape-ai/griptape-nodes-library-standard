---
description: A group that reruns the nodes inside it when they fail, up to a set number of attempts.
when_to_use: Use it when a step can fail now and then, such as a network call, and you want it tried again automatically.
---

## When to use it

Put the nodes that might fail inside the group. Wire their success output to the group's **Succeeded** input and their failure output to **Failed**. If Failed fires and attempts remain, the whole group runs again. Otherwise the flow continues downstream.

## Parameters

### max_iterations

How many retries to allow after the first attempt. The default is 3, so the group can run up to 4 times. A value of 0 runs it once with no retries.

### iteration

The current attempt number, starting at 0. Connect it to nodes inside the group if they need to know.

### raise_on_failure

Off by default. When on, the workflow stops with an error if every attempt fails. When off, the flow continues and you can check `was_successful`.

### was_successful

True if the group ended through **Succeeded**. False if the retries ran out.

### total_iterations

How many attempts ran.
