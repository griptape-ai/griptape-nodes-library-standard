- Rename File and Move Files with `overwrite` on no longer delete the existing destination before
  the move. If the move fails, the destination is left as it was. Move Files' description no longer
  says it copies and deletes, and its always-empty `copied_paths` output is gone; use `moved_paths`.
  A saved workflow that connected `copied_paths` loses that connection when it opens.
  [#1031](https://github.com/griptape-ai/griptape-nodes-library-standard/issues/1031)
