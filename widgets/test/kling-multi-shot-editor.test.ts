import { join } from "node:path";

import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";

import type { KlingMultiShotEditorProps, KlingShot } from "../src/KlingMultiShotEditor/KlingMultiShotEditor";
import type { WidgetMount } from "@griptape-ai/nodes-widget-api";
import { buildIntoScratch, importMount, removeDir } from "./support/bundle";
import { type MountedWidget, flush, mountWidget, widgetProps } from "./support/widget";

type Mount = WidgetMount<KlingMultiShotEditorProps>;
type Props = Partial<KlingMultiShotEditorProps>;

// Each behaviour runs against the TypeScript source and against the bundle the build emits,
// so the bundle is held to the same contract the editor relies on.
const implementations: Array<[string, () => Promise<Mount>]> = [
  [
    "source",
    async () => (await import("../src/KlingMultiShotEditor/KlingMultiShotEditor")).default as Mount,
  ],
  [
    "bundle",
    async () => {
      const { outDir } = await buildIntoScratch(undefined, ["KlingMultiShotEditor"]);
      bundleDirs.push(outDir);
      return importMount<KlingMultiShotEditorProps>(join(outDir, "KlingMultiShotEditor.js"));
    },
  ],
];
const bundleDirs: string[] = [];

afterAll(async () => {
  await Promise.all(bundleDirs.map(removeDir));
});

function shots(...durations: number[]): KlingShot[] {
  return durations.map((duration, i) => ({ name: `Shot${i + 1}`, duration, description: "" }));
}

function down(target: Element, init: PointerEventInit = {}): void {
  flush(() => {
    target.dispatchEvent(new PointerEvent("pointerdown", { bubbles: true, cancelable: true, ...init }));
  });
}

// Fires an event on the target and lets the editor render what it changed.
function fire(target: EventTarget, event: Event): void {
  flush(() => {
    target.dispatchEvent(event);
  });
}

// Leaving a field: React listens for the bubbling focusout the browser sends with blur.
const blur = () => new FocusEvent("focusout", { bubbles: true });

describe.each(implementations)("KlingMultiShotEditor (%s)", (_label, load) => {
  let KlingMultiShotEditor: Mount;
  let container: HTMLElement;
  let handles: MountedWidget<KlingMultiShotEditorProps>[];

  beforeAll(async () => {
    KlingMultiShotEditor = await load();
  });

  beforeEach(() => {
    container = document.createElement("div");
    document.body.appendChild(container);
    handles = [];
  });

  afterEach(async () => {
    for (const h of handles) await h.cleanup();
    container.remove();
    document.body.innerHTML = "";
    vi.restoreAllMocks();
  });

  // Mounts with the editor's props filled in around the ones a test cares about.
  function mount(props: Props) {
    const onChange = vi.fn<(value: KlingShot[]) => void>();
    const mounted = mountWidget(KlingMultiShotEditor, container, widgetProps({ onChange, ...props }));
    handles.push(mounted);
    return { ...mounted, onChange: (props.onChange as typeof onChange | undefined) ?? onChange };
  }

  const items = () => [...container.querySelectorAll<HTMLElement>(".shot-item")];
  const badges = () => items().map((i) => i.querySelector(":scope > div > span:nth-child(2)")?.textContent);
  const durations = () => items().map((i) => i.querySelector(":scope > div > div > span:nth-child(2)")?.textContent);
  const status = () => [...container.querySelectorAll(".multi-shot-container > div:nth-last-child(3) > span")].map((s) => s.textContent);
  const addButton = () => {
    const btn = [...container.querySelectorAll<HTMLElement>(".multi-shot-container > div")].find((d) => d.textContent?.trim() === "+ Shot");
    if (!btn) throw new Error("no add button");
    return btn;
  };
  const stepper = (index: number, label: "+" | "-") => {
    const btn = [...(items()[index]?.querySelectorAll<HTMLElement>(":scope > div > div > span") ?? [])].find((s) => s.textContent === label);
    if (!btn) throw new Error(`no ${label} on shot ${index}`);
    return btn;
  };
  const trash = (index: number) => items()[index]?.querySelector("svg")?.parentElement ?? null;
  const textarea = (index: number) => {
    const ta = items()[index]?.querySelector("textarea");
    if (!ta) throw new Error(`no textarea on shot ${index}`);
    return ta;
  };
  const counter = (index: number) => items()[index]?.querySelector(":scope > div:last-child")?.textContent;
  const lastCommit = (onChange: ReturnType<typeof vi.fn>) => onChange.mock.lastCall?.[0] as KlingShot[];

  describe("mount", () => {
    it("returns a handle and builds the editor inside the container", () => {
      const handle = mount({ value: shots(3, 2) });
      expect(handle.result).toEqual({ update: expect.any(Function), cleanup: expect.any(Function) });
      expect(container.querySelectorAll(".multi-shot-container")).toHaveLength(1);
      expect(badges()).toEqual(["Shot1", "Shot2"]);
      expect(durations()).toEqual(["3 s", "2 s"]);
      expect(status()).toEqual(["2 / 6 shots", "5s (3-15s)"]);
      expect(container.textContent).toContain("v0.6.1");
    });

    it("starts with one 3 s shot when the value is empty", () => {
      mount({ value: [] });
      expect(badges()).toEqual(["Shot1"]);
      expect(durations()).toEqual(["3 s"]);
    });

    it("starts with one 3 s shot when the value is missing", () => {
      mount({ value: null });
      expect(badges()).toEqual(["Shot1"]);
    });

    it("shows each shot's description and its length", () => {
      mount({ value: [{ name: "Shot1", duration: 3, description: "a cat" }] });
      expect(textarea(0).value).toBe("a cat");
      expect(textarea(0).maxLength).toBe(512);
      expect(counter(0)).toBe("5 / 512");
    });

    it("leaves the incoming value unmodified", () => {
      const value = shots(3);
      mount({ value });
      expect(value[0]).toEqual({ name: "Shot1", duration: 3, description: "" });
    });
  });

  describe("update", () => {
    it("keeps the shots it mounted with on update: a fresh mount applies a new value", async () => {
      // Given a mounted editor
      const first = mount({ value: shots(3) });
      first.update(widgetProps({ value: shots(4, 4) }));
      expect(badges()).toEqual(["Shot1"]);

      // When the editor is cleaned up and mounts again with a new value
      await first.cleanup();
      mount({ value: shots(4, 4) });

      // Then the container holds just the new editor
      expect(container.querySelectorAll(".multi-shot-container")).toHaveLength(1);
      expect(durations()).toEqual(["4 s", "4 s"]);
    });
  });

  describe("duration stepper", () => {
    it("commits a longer shot on +", () => {
      const { onChange } = mount({ value: shots(3, 2) });
      down(stepper(1, "+"));
      expect(onChange).toHaveBeenCalledTimes(1);
      expect(lastCommit(onChange).map((s) => s.duration)).toEqual([3, 3]);
      expect(durations()).toEqual(["3 s", "3 s"]);
    });

    it("commits a shorter shot on -", () => {
      const { onChange } = mount({ value: shots(3, 2) });
      down(stepper(0, "-"));
      expect(lastCommit(onChange).map((s) => s.duration)).toEqual([2, 2]);
    });

    it("commits shots with ids, as copies", () => {
      const { onChange } = mount({ value: shots(3, 2) });
      down(stepper(0, "+"));
      const committed = lastCommit(onChange);
      expect(committed).toEqual([
        { id: "shot-1", name: "Shot1", duration: 4, description: "" },
        { id: "shot-2", name: "Shot2", duration: 2, description: "" },
      ]);
      committed[0]!.duration = 99;
      expect(durations()).toEqual(["4 s", "2 s"]);
    });

    it("blocks + when the 15 s total is reached", () => {
      const { onChange } = mount({ value: shots(10, 5) });
      down(stepper(0, "+"));
      expect(onChange).not.toHaveBeenCalled();
      expect(stepper(0, "+").style.cursor).toBe("default");
    });

    it("drops the hover colour of a + that re-enables after the pointer left it", () => {
      // Given a + hovered until the budget runs out, then left
      mount({ value: shots(10, 4) });
      fire(stepper(0, "+"), new PointerEvent("pointerover", { bubbles: true }));
      down(stepper(0, "+"));
      expect(stepper(0, "+").style.cursor).toBe("default");
      fire(stepper(0, "+"), new PointerEvent("pointerout", { bubbles: true }));

      // When - frees the budget again
      down(stepper(0, "-"));

      // Then + is enabled without the hover colour
      expect(stepper(0, "+").style.cursor).toBe("pointer");
      expect(stepper(0, "+").style.background).toBe("transparent");
    });

    it("drops the hover colour of + Shot that re-enables after the pointer left it", () => {
      // Given + Shot hovered until six shots, then left
      mount({ value: shots(2, 2, 2, 2, 2) });
      fire(addButton(), new PointerEvent("pointerover", { bubbles: true }));
      down(addButton());
      expect(addButton().style.cursor).toBe("not-allowed");
      fire(addButton(), new PointerEvent("pointerout", { bubbles: true }));

      // When a shot is removed
      down(trash(0)!);

      // Then + Shot is enabled without the hover colour
      expect(addButton().style.cursor).toBe("pointer");
      expect(addButton().style.background).toBe("#252525");
    });

    it("blocks - when the total would drop below 3 s", () => {
      const { onChange } = mount({ value: shots(2, 1) });
      down(stepper(0, "-"));
      expect(onChange).not.toHaveBeenCalled();
    });
  });

  describe("adding and removing shots", () => {
    it("adds a 2 s shot", () => {
      const { onChange } = mount({ value: shots(3, 2) });
      down(addButton());
      expect(lastCommit(onChange)).toHaveLength(3);
      expect(lastCommit(onChange)[2]).toEqual({ id: "shot-3", name: "Shot3", duration: 2, description: "" });
      expect(badges()).toEqual(["Shot1", "Shot2", "Shot3"]);
    });

    it("continues numbering ids after the highest existing one, and fits the new shot into what's left", () => {
      const { onChange } = mount({ value: [{ id: "shot-5", name: "Shot1", duration: 14, description: "" }] });
      down(addButton());
      expect(lastCommit(onChange)[1]).toEqual({ id: "shot-6", name: "Shot2", duration: 1, description: "" });
    });

    it("blocks adding at six shots", () => {
      const { onChange } = mount({ value: shots(2, 2, 2, 2, 2, 2) });
      down(addButton());
      expect(onChange).not.toHaveBeenCalled();
      expect(addButton().style.cursor).toBe("not-allowed");
    });

    it("removes a shot and renumbers the rest", () => {
      const { onChange } = mount({ value: [
        { name: "Shot1", duration: 3, description: "a" },
        { name: "Shot2", duration: 2, description: "b" },
        { name: "Shot3", duration: 2, description: "c" },
      ] });
      down(trash(0)!);
      expect(lastCommit(onChange).map((s) => [s.name, s.description])).toEqual([["Shot1", "b"], ["Shot2", "c"]]);
      expect(badges()).toEqual(["Shot1", "Shot2"]);
    });

    it("tops the last shot up to the 3 s minimum after a removal", () => {
      const { onChange } = mount({ value: shots(2, 1) });
      down(trash(0)!);
      expect(lastCommit(onChange).map((s) => s.duration)).toEqual([3]);
    });

    it("keeps the last shot", () => {
      const { onChange } = mount({ value: shots(3) });
      down(trash(0)!);
      expect(onChange).not.toHaveBeenCalled();
      expect(items()).toHaveLength(1);
    });
  });

  describe("description", () => {
    it("updates the counter while typing and commits on blur", () => {
      // Given a shot's description field
      const { onChange } = mount({ value: shots(3) });
      const ta = textarea(0);

      // When the user types
      ta.value = "hello";
      fire(ta, new Event("input", { bubbles: true }));

      // Then the counter follows without committing
      expect(counter(0)).toBe("5 / 512");
      expect(onChange).not.toHaveBeenCalled();

      // And leaving the field commits the text
      fire(ta, blur());
      expect(lastCommit(onChange)[0]?.description).toBe("hello");
    });

    it("keeps key presses from reaching the node", () => {
      mount({ value: shots(3) });
      const onKey = vi.fn();
      container.addEventListener("keydown", onKey);
      textarea(0).dispatchEvent(new KeyboardEvent("keydown", { key: "Delete", bubbles: true }));
      expect(onKey).not.toHaveBeenCalled();
    });
  });

  describe("drag to reorder", () => {
    // Lay shot items out 100 px tall, one under the other.
    beforeEach(() => {
      vi.spyOn(Element.prototype, "getBoundingClientRect").mockImplementation(function (this: Element) {
        const index = Number((this as HTMLElement).dataset?.index ?? 0);
        return new DOMRect(0, index * 100, 300, 100);
      });
    });

    const handleOf = (index: number) => items()[index]!.querySelector<HTMLElement>(".drag-handle")!;
    const move = (y: number) => fire(document, new PointerEvent("pointermove", { clientY: y, cancelable: true }));
    const up = () => fire(document, new PointerEvent("pointerup", {}));

    it("moves a shot below another and commits the new order", () => {
      // Given three shots
      const { onChange } = mount({ value: [
        { name: "Shot1", duration: 3, description: "a" },
        { name: "Shot2", duration: 2, description: "b" },
        { name: "Shot3", duration: 2, description: "c" },
      ] });

      // When the first is dragged past the middle of the last
      down(handleOf(0), { clientY: 10 });
      expect((document.body.lastElementChild as HTMLElement | null)?.style.position).toBe("fixed");
      move(260);
      up();

      // Then it lands at the end, the shots are renumbered, and the clone is gone
      expect(lastCommit(onChange).map((s) => [s.name, s.description])).toEqual([["Shot1", "b"], ["Shot2", "c"], ["Shot3", "a"]]);
      expect(badges()).toEqual(["Shot1", "Shot2", "Shot3"]);
      expect([...document.body.children].filter((c) => (c as HTMLElement).style.position === "fixed")).toHaveLength(0);
    });

    it("marks the drop target while dragging", () => {
      mount({ value: shots(3, 2, 2) });
      down(handleOf(2), { clientY: 210 });
      move(40);
      expect(items()[0]!.style.borderTop).toBe("2px solid #4a9eff");
      up();
    });

    it("commits nothing when the shot is dropped where it was", () => {
      const { onChange } = mount({ value: shots(3, 2) });
      down(handleOf(0), { clientY: 10 });
      move(20);
      up();
      expect(onChange).not.toHaveBeenCalled();
    });
  });

  describe("disabled", () => {
    it("offers no editing controls and commits nothing", () => {
      // Given a read-only node
      const { onChange } = mount({ value: shots(3, 2), disabled: true });

      // Then there is no trash or drag affordance and the field is read-only
      expect(trash(0)).toBeNull();
      expect(items()[0]!.querySelector<HTMLElement>(".drag-handle")!.style.cursor).toBe("default");
      expect(textarea(0).disabled).toBe(true);
      expect(addButton().style.cursor).toBe("not-allowed");

      // And pressing the controls changes nothing
      down(stepper(0, "+"));
      down(addButton());
      down(items()[0]!.querySelector(".drag-handle")!);
      expect(document.body.querySelectorAll(".shot-item")).toHaveLength(2);
      fire(textarea(0), blur());
      expect(onChange).not.toHaveBeenCalled();
    });
  });

  describe("cleanup", () => {
    it("removes a drag in progress and stops listening on the document", async () => {
      // Given a drag under way
      vi.spyOn(Element.prototype, "getBoundingClientRect").mockImplementation(function (this: Element) {
        return new DOMRect(0, Number((this as HTMLElement).dataset?.index ?? 0) * 100, 300, 100);
      });
      const { cleanup, onChange } = mount({ value: shots(3, 2) });
      down(items()[0]!.querySelector(".drag-handle")!, { clientY: 10 });
      expect(document.body.querySelectorAll(".shot-item")).toHaveLength(3);
      const removeSpy = vi.spyOn(document, "removeEventListener");

      // When the widget is cleaned up
      await cleanup();

      // Then the editor and its clone are gone and a later drop does nothing
      expect(removeSpy).toHaveBeenCalledWith("pointermove", expect.any(Function));
      expect(removeSpy).toHaveBeenCalledWith("pointerup", expect.any(Function));
      expect(document.body.querySelectorAll(".shot-item")).toHaveLength(0);
      fire(document, new PointerEvent("pointermove", { clientY: 190 }));
      fire(document, new PointerEvent("pointerup", {}));
      expect(onChange).not.toHaveBeenCalled();
    });
  });
});
