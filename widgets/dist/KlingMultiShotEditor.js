// src/KlingMultiShotEditor/KlingMultiShotEditor.ts
var WIDGET_VERSION = "0.6.1";
function MultiShotEditor(container, props) {
  const { value, onChange, disabled } = props;
  let nextShotId = 1;
  function assignId(shot) {
    if (!shot.id) {
      shot.id = `shot-${nextShotId++}`;
    } else {
      const num = parseInt(shot.id.replace("shot-", ""), 10);
      if (!isNaN(num) && num >= nextShotId) {
        nextShotId = num + 1;
      }
    }
    return shot;
  }
  let shots = Array.isArray(value) && value.length > 0 ? value.map((s) => assignId({ ...s })) : [assignId({ name: "Shot1", duration: 3, description: "" })];
  let dragSourceIndex = null;
  let dragOverIndex = null;
  const MAX_SHOTS = 6;
  const MIN_SHOTS = 1;
  const MAX_DESCRIPTION_LENGTH = 512;
  const MAX_TOTAL_DURATION = 15;
  const MIN_TOTAL_DURATION = 3;
  const MIN_DURATION = 1;
  const MAX_DURATION = 15;
  const PLACEHOLDER = "Describe the shot information, such as: who is where and what is happening.";
  function renumberShots() {
    shots.forEach((shot, i) => {
      shot.name = `Shot${i + 1}`;
    });
  }
  function totalDuration() {
    return shots.reduce((sum, s) => sum + (s.duration || 0), 0);
  }
  function durationBudgetFor(index) {
    const othersTotal = shots.reduce(
      (sum, s, i) => sum + (i === index ? 0 : s.duration || 0),
      0
    );
    return MAX_TOTAL_DURATION - othersTotal;
  }
  function emitChange() {
    if (disabled || !onChange) return;
    onChange(shots.map((s) => ({ ...s })));
  }
  function render() {
    container.innerHTML = "";
    const wrapper = el("div", {
      className: "multi-shot-container nodrag nowheel",
      style: `
        display: flex;
        flex-direction: column;
        gap: 0;
        padding: 8px;
        background-color: #1a1a1a;
        border-radius: 6px;
        user-select: none;
        box-sizing: border-box;
        width: 100%;
        height: 100%;
        max-height: 100%;
        min-height: 0;
        overflow-y: auto;
        overflow-x: hidden;
        overscroll-behavior: contain;
      `
    });
    shots.forEach((shot, index) => {
      wrapper.appendChild(buildShotItem(shot, index));
    });
    wrapper.appendChild(buildStatusBar());
    wrapper.appendChild(buildAddButton());
    wrapper.appendChild(buildVersionLabel());
    container.appendChild(wrapper);
  }
  function buildShotItem(shot, index) {
    const item = el("div", {
      className: "shot-item",
      "data-index": index,
      style: `
        display: flex;
        flex-direction: column;
        gap: 0;
        border-bottom: 1px solid #2a2a2a;
        padding-bottom: 8px;
        margin-bottom: 8px;
        ${dragOverIndex === index ? "border-top: 2px solid #4a9eff;" : ""}
      `
    });
    const header = el("div", {
      style: `
        display: flex;
        align-items: center;
        gap: 8px;
        height: 32px;
      `
    });
    const handle = el("span", {
      className: "drag-handle",
      textContent: "|||",
      style: `
        font-size: 18px;
        color: #666;
        cursor: ${disabled ? "default" : "grab"};
        flex-shrink: 0;
        width: 20px;
        text-align: center;
        line-height: 1;
      `
    });
    if (!disabled) {
      handle.addEventListener("pointerdown", (e) => onDragStart(e, index));
    }
    header.appendChild(handle);
    header.appendChild(
      el("span", {
        textContent: shot.name,
        style: `
          background: #333;
          color: #ddd;
          font-size: 12px;
          font-weight: 600;
          padding: 3px 10px;
          border-radius: 4px;
          white-space: nowrap;
        `
      })
    );
    const budget = durationBudgetFor(index);
    const wouldGoBelow = totalDuration() - 1 < MIN_TOTAL_DURATION;
    const canIncrease = !disabled && shot.duration < MAX_DURATION && shot.duration < budget;
    const canDecrease = !disabled && shot.duration > MIN_DURATION && !wouldGoBelow;
    const stepper = el("div", {
      style: `
        display: flex;
        align-items: center;
        gap: 0;
        background: #333;
        border-radius: 4px;
        overflow: hidden;
      `
    });
    const decBtn = buildStepperButton("-", canDecrease, (e) => {
      e.stopPropagation();
      shots[index].duration = Math.max(MIN_DURATION, shots[index].duration - 1);
      emitChange();
      render();
    });
    const durationLabel = el("span", {
      textContent: `${shot.duration} s`,
      style: `
        color: #ddd;
        font-size: 12px;
        padding: 2px 8px;
        min-width: 32px;
        text-align: center;
        white-space: nowrap;
      `
    });
    const incBtn = buildStepperButton("+", canIncrease, (e) => {
      e.stopPropagation();
      const maxAllowed = Math.min(MAX_DURATION, budget);
      shots[index].duration = Math.min(maxAllowed, shots[index].duration + 1);
      emitChange();
      render();
    });
    stepper.appendChild(incBtn);
    stepper.appendChild(durationLabel);
    stepper.appendChild(decBtn);
    header.appendChild(stepper);
    header.appendChild(el("div", { style: "flex: 1;" }));
    if (!disabled) {
      const trash = el("span", {
        innerHTML: trashSVG(),
        style: `
          cursor: pointer;
          flex-shrink: 0;
          display: flex;
          align-items: center;
          opacity: 0.5;
        `
      });
      trash.addEventListener("pointerenter", () => {
        trash.style.opacity = "1";
      });
      trash.addEventListener("pointerleave", () => {
        trash.style.opacity = "0.5";
      });
      trash.addEventListener("pointerdown", (e) => {
        e.stopPropagation();
        if (shots.length <= 1) return;
        shots.splice(index, 1);
        renumberShots();
        const total = totalDuration();
        if (total < MIN_TOTAL_DURATION) {
          const lastShot = shots[shots.length - 1];
          lastShot.duration += MIN_TOTAL_DURATION - total;
        }
        emitChange();
        render();
      });
      header.appendChild(trash);
    }
    item.appendChild(header);
    const descLength = (shot.description || "").length;
    const textarea = el("textarea", {
      placeholder: PLACEHOLDER,
      value: shot.description || "",
      disabled,
      style: `
        width: 100%;
        min-height: 72px;
        margin-top: 4px;
        padding: 8px 10px;
        background: #252525;
        border: 1px solid ${descLength >= MAX_DESCRIPTION_LENGTH ? "#c44" : "#333"};
        border-radius: 6px;
        color: #ccc;
        font-size: 12px;
        font-family: inherit;
        resize: vertical;
        box-sizing: border-box;
        outline: none;
        user-select: text;
        -webkit-user-select: text;
      `
    });
    textarea.maxLength = MAX_DESCRIPTION_LENGTH;
    textarea.addEventListener("focus", () => {
      const len = shots[index].description.length;
      textarea.style.borderColor = len >= MAX_DESCRIPTION_LENGTH ? "#c44" : "#555";
    });
    textarea.addEventListener("blur", () => {
      shots[index].description = textarea.value;
      const len = textarea.value.length;
      textarea.style.borderColor = len >= MAX_DESCRIPTION_LENGTH ? "#c44" : "#333";
      counter.textContent = `${len} / ${MAX_DESCRIPTION_LENGTH}`;
      counter.style.color = len >= MAX_DESCRIPTION_LENGTH ? "#c44" : "#555";
      emitChange();
    });
    textarea.addEventListener("input", (e) => {
      const target = e.target;
      shots[index].description = target.value;
      counter.textContent = `${target.value.length} / ${MAX_DESCRIPTION_LENGTH}`;
      counter.style.color = target.value.length >= MAX_DESCRIPTION_LENGTH ? "#c44" : "#555";
      textarea.style.borderColor = target.value.length >= MAX_DESCRIPTION_LENGTH ? "#c44" : "#555";
    });
    textarea.addEventListener("pointerdown", (e) => e.stopPropagation());
    textarea.addEventListener("mousedown", (e) => e.stopPropagation());
    textarea.addEventListener("keydown", (e) => e.stopPropagation());
    item.appendChild(textarea);
    const counter = el("div", {
      textContent: `${descLength} / ${MAX_DESCRIPTION_LENGTH}`,
      style: `
        font-size: 10px;
        color: ${descLength >= MAX_DESCRIPTION_LENGTH ? "#c44" : "#555"};
        text-align: right;
        margin-top: 2px;
      `
    });
    item.appendChild(counter);
    return item;
  }
  function buildStepperButton(label, enabled, onClick) {
    const btn = el("span", {
      textContent: label,
      style: `
        display: flex;
        align-items: center;
        justify-content: center;
        width: 22px;
        height: 22px;
        font-size: 10px;
        color: ${enabled ? "#ccc" : "#555"};
        cursor: ${enabled ? "pointer" : "default"};
        user-select: none;
        opacity: ${enabled ? 1 : 0.4};
      `
    });
    if (enabled) {
      btn.addEventListener("pointerenter", () => {
        btn.style.background = "#444";
      });
      btn.addEventListener("pointerleave", () => {
        btn.style.background = "transparent";
      });
      btn.addEventListener("pointerdown", onClick);
    } else {
      btn.addEventListener("pointerdown", (e) => e.stopPropagation());
    }
    return btn;
  }
  function buildStatusBar() {
    const total = totalDuration();
    const overBudget = total > MAX_TOTAL_DURATION;
    const underBudget = total < MIN_TOTAL_DURATION;
    const durationColor = overBudget || underBudget ? "#c44" : "#666";
    const bar = el("div", {
      style: `
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 10px;
        color: #666;
        padding: 4px 2px;
        margin-bottom: 2px;
      `
    });
    bar.appendChild(
      el("span", {
        textContent: `${shots.length} / ${MAX_SHOTS} shots`
      })
    );
    bar.appendChild(
      el("span", {
        textContent: `${total}s (${MIN_TOTAL_DURATION}-${MAX_TOTAL_DURATION}s)`,
        style: `color: ${durationColor};`
      })
    );
    return bar;
  }
  function buildAddButton() {
    const atMaxShots = shots.length >= MAX_SHOTS;
    const remaining = MAX_TOTAL_DURATION - totalDuration();
    const cannotFitAnother = remaining < 1;
    const addDisabled = disabled || atMaxShots || cannotFitAnother;
    const btn = el("div", {
      style: `
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 5px 12px;
        margin-top: 4px;
        background: #252525;
        border: 1px solid #333;
        border-radius: 6px;
        color: #ccc;
        font-size: 12px;
        cursor: ${addDisabled ? "not-allowed" : "pointer"};
        opacity: ${addDisabled ? 0.4 : 1};
        align-self: flex-start;
        user-select: none;
      `
    });
    btn.innerHTML = `<span style="font-size:14px;">+</span> Shot`;
    if (!addDisabled) {
      btn.addEventListener("pointerenter", () => {
        btn.style.background = "#333";
      });
      btn.addEventListener("pointerleave", () => {
        btn.style.background = "#252525";
      });
      btn.addEventListener("pointerdown", (e) => {
        e.stopPropagation();
        const defaultDur = Math.min(2, remaining);
        shots.push(
          assignId({
            name: `Shot${shots.length + 1}`,
            duration: defaultDur,
            description: ""
          })
        );
        emitChange();
        render();
      });
    }
    return btn;
  }
  function buildVersionLabel() {
    return el("div", {
      textContent: `v${WIDGET_VERSION}`,
      style: `
        font-size: 9px;
        color: #444;
        text-align: right;
        margin-top: 4px;
      `
    });
  }
  let dragClone = null;
  let dragOffsetY = 0;
  let shotItems = [];
  function onDragStart(e, index) {
    if (disabled) return;
    e.preventDefault();
    e.stopPropagation();
    dragSourceIndex = index;
    dragOverIndex = null;
    const itemEl = e.target.closest(".shot-item");
    if (!itemEl) return;
    const rect = itemEl.getBoundingClientRect();
    dragOffsetY = e.clientY - rect.top;
    dragClone = itemEl.cloneNode(true);
    dragClone.style.position = "fixed";
    dragClone.style.left = `${rect.left}px`;
    dragClone.style.top = `${rect.top}px`;
    dragClone.style.width = `${rect.width}px`;
    dragClone.style.opacity = "0.85";
    dragClone.style.pointerEvents = "none";
    dragClone.style.zIndex = "9999";
    dragClone.style.boxShadow = "0 4px 16px rgba(0,0,0,0.6)";
    dragClone.style.background = "#1a1a1a";
    dragClone.style.borderRadius = "6px";
    document.body.appendChild(dragClone);
    itemEl.style.opacity = "0.3";
    shotItems = Array.from(container.querySelectorAll(".shot-item")).map(
      (el2) => ({
        el: el2,
        rect: el2.getBoundingClientRect(),
        index: Number(el2.dataset.index)
      })
    );
    document.addEventListener("pointermove", onDragMove);
    document.addEventListener("pointerup", onDragEnd);
  }
  function onDragMove(e) {
    if (dragSourceIndex === null || !dragClone) return;
    e.preventDefault();
    dragClone.style.top = `${e.clientY - dragOffsetY}px`;
    let newOverIndex = null;
    for (const item of shotItems) {
      const midY = item.rect.top + item.rect.height / 2;
      if (e.clientY < midY) {
        newOverIndex = item.index;
        break;
      }
    }
    if (newOverIndex === null) {
      newOverIndex = shots.length;
    }
    if (newOverIndex !== dragOverIndex) {
      dragOverIndex = newOverIndex;
      const lastIndex = shots.length - 1;
      shotItems.forEach((item) => {
        item.el.style.borderTop = "none";
        item.el.style.borderBottom = "1px solid #2a2a2a";
        if (item.index === dragSourceIndex) return;
        if (dragOverIndex === shots.length && item.index === lastIndex) {
          item.el.style.borderBottom = "2px solid #4a9eff";
        } else if (item.index === dragOverIndex) {
          item.el.style.borderTop = "2px solid #4a9eff";
        }
      });
    }
  }
  function onDragEnd(_e) {
    document.removeEventListener("pointermove", onDragMove);
    document.removeEventListener("pointerup", onDragEnd);
    if (dragClone) {
      dragClone.remove();
      dragClone = null;
    }
    if (dragSourceIndex !== null && dragOverIndex !== null && dragOverIndex !== dragSourceIndex) {
      const [moved] = shots.splice(dragSourceIndex, 1);
      const insertAt = dragOverIndex > dragSourceIndex ? dragOverIndex - 1 : dragOverIndex;
      shots.splice(insertAt, 0, moved);
      renumberShots();
      emitChange();
    }
    dragSourceIndex = null;
    dragOverIndex = null;
    shotItems = [];
    render();
  }
  function el(tag, attrs) {
    const element = document.createElement(tag);
    if (attrs) {
      Object.entries(attrs).forEach(([key, val]) => {
        if (key === "style") {
          element.setAttribute("style", String(val));
        } else if (key === "textContent") {
          element.textContent = String(val);
        } else if (key === "innerHTML") {
          element.innerHTML = String(val);
        } else if (key === "className") {
          element.className = String(val);
        } else if (key === "disabled") {
          element.disabled = !!val;
        } else if (key === "placeholder") {
          element.placeholder = String(val);
        } else if (key === "value") {
          element.value = String(val);
        } else {
          element.setAttribute(key, String(val));
        }
      });
    }
    return element;
  }
  function trashSVG() {
    return `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#888" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <polyline points="3 6 5 6 21 6"/>
      <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
    </svg>`;
  }
  render();
  return () => {
    document.removeEventListener("pointermove", onDragMove);
    document.removeEventListener("pointerup", onDragEnd);
    if (dragClone) {
      dragClone.remove();
      dragClone = null;
    }
  };
}
export {
  MultiShotEditor as default
};
