- `remove_blank` and `remove_duplicates` on Create List, Create Image List and the other typed
  create list nodes work on items that aren't text. `remove_blank` drops missing items and blank
  text, and `remove_duplicates` keeps the first copy of each item in its original order. Images
  count as duplicates when they share a URL.
  [#1032](https://github.com/griptape-ai/griptape-nodes-library-standard/issues/1032)
