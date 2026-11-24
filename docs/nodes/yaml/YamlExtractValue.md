---
description: Pulls a value or a list of values out of YAML using a JMESPath expression.
when_to_use: Use it when you need one field, or the same field from every item, out of a larger YAML document.
---

## When to use it

You have YAML and only need part of it, such as a name, the first item, or every title in a list. Enter a JMESPath expression in `path` and the node returns the match. The result updates as soon as `yaml` or `path` changes.

## Example

An asset manifest, as saved by a publish script:

```yaml
asset: hero_ship
version: v012
lods:
  - name: high
    polys: 2400000
  - name: low
    polys: 18000
textures:
  - name: hull_diffuse
    res: 8192
  - name: hull_normal
    res: 4096
```

| `path` | What it asks for | Output |
| --- | --- | --- |
| `version` | A single value | `v012` |
| `lods[0]` | The first LOD, as a mapping | `name: high` and `polys: 2400000` on two lines |
| `lods[*].name` | The name of every LOD | `- high` and `- low` on two lines |
| `textures[*].res` | The resolution of every texture | `- 8192` and `- 4096` on two lines |
| ``textures[?res >= `8192`].name`` | Only the textures at 8K or above | `- hull_diffuse` |
| `materials` | A key that isn't there | `{}` |

The output is always text, so a list comes back as YAML text, not as a list. If you need a real list to connect to a list node, use [JSON Extract Value](../json/JsonExtractValue.md) on JSON data instead.


## Parameters

### path

A JMESPath expression.

| Expression | Result |
| --- | --- |
| `user.name` | A nested key |
| `items[0].title` | A field of the first list item |
| `[*].name` | The `name` of every item in a top-level list |

If `path` is empty, the whole document is returned.

### output

Always text. A mapping or list is written back out as YAML. A single value such as a string or number is returned as its text.

If the expression matches nothing, the output is `{}`. If `yaml` is empty, the output is also `{}`.

> [!NOTE]
> An invalid YAML document or an invalid JMESPath expression raises an error that names the field.
