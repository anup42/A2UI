import React, { useMemo, useState } from "https://esm.sh/react@18.3.1";
import { createRoot } from "https://esm.sh/react-dom@18.3.1/client";
import {
  JSONUIProvider,
  Renderer,
  useBoundProp,
} from "https://esm.sh/@json-render/react@0.16.0";

const rootEl = document.getElementById("root");
const fallbackEl = document.getElementById("fallback");

function asText(value) {
  if (value === null || value === undefined) return "";
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  try {
    return JSON.stringify(value);
  } catch (_err) {
    return String(value);
  }
}

function cn(...names) {
  return names.filter(Boolean).join(" ");
}

function openExternal(url) {
  if (typeof url !== "string" || !url.trim()) return;
  window.open(url, "_blank", "noopener,noreferrer");
}

const styles = {
  screen: {
    display: "flex",
    flexDirection: "column",
    gap: "10px",
    minHeight: "100%",
    padding: "14px",
    background: "linear-gradient(180deg,#ffffff 0%,#f9fbff 100%)",
  },
  row: {
    display: "flex",
    alignItems: "stretch",
    gap: "8px",
    flexWrap: "wrap",
  },
  card: {
    border: "1px solid #d9e0ea",
    borderRadius: "14px",
    padding: "12px",
    background: "#ffffff",
    boxShadow: "0 3px 12px rgba(17,24,39,0.06)",
  },
};

function Column({ children }) {
  return React.createElement("div", { style: styles.screen }, children);
}

function Row({ props, children }) {
  const justifyMap = {
    start: "flex-start",
    end: "flex-end",
    center: "center",
    between: "space-between",
    around: "space-around",
    evenly: "space-evenly",
  };
  const justify = justifyMap[String(props?.justify || "").toLowerCase()] || "flex-start";
  return React.createElement(
    "div",
    {
      style: {
        ...styles.row,
        justifyContent: justify,
      },
    },
    children,
  );
}

function List({ children }) {
  return React.createElement(
    "div",
    {
      style: {
        display: "flex",
        flexDirection: "column",
        gap: "10px",
      },
    },
    children,
  );
}

function Card({ children }) {
  return React.createElement("section", { style: styles.card }, children);
}

function Text({ props }) {
  const variant = String(props?.variant || "body").toLowerCase();
  const text = asText(props?.text);
  const variantStyle = {
    h1: { fontSize: "24px", fontWeight: 700, lineHeight: "1.2" },
    h2: { fontSize: "20px", fontWeight: 700, lineHeight: "1.25" },
    h3: { fontSize: "17px", fontWeight: 600, lineHeight: "1.3" },
    body: { fontSize: "14px", fontWeight: 400, lineHeight: "1.45" },
    caption: { fontSize: "12px", fontWeight: 400, lineHeight: "1.35", color: "#4b5563" },
    chip: {
      fontSize: "11px",
      fontWeight: 600,
      lineHeight: "1.2",
      border: "1px solid #d4d8e1",
      background: "#f5f7fb",
      borderRadius: "999px",
      padding: "3px 8px",
      display: "inline-block",
      width: "fit-content",
    },
    label: { fontSize: "13px", fontWeight: 600, lineHeight: "1.3", color: "#334155" },
  };
  return React.createElement(
    "p",
    {
      style: {
        margin: 0,
        whiteSpace: "pre-wrap",
        wordBreak: "break-word",
        ...(variantStyle[variant] || variantStyle.body),
      },
    },
    text,
  );
}

function Image({ props }) {
  const url = asText(props?.url);
  if (!url) {
    return null;
  }
  return React.createElement("img", {
    src: url,
    alt: asText(props?.alt || "image"),
    style: {
      width: "100%",
      maxHeight: "240px",
      objectFit: String(props?.fit || "cover").toLowerCase() === "contain" ? "contain" : "cover",
      borderRadius: "12px",
      border: "1px solid #dce1ea",
      background: "#f8fafc",
    },
  });
}

function Icon({ props }) {
  const iconUrl = asText(props?.name);
  if (!iconUrl) return null;
  return React.createElement("img", {
    src: iconUrl,
    alt: "icon",
    style: {
      width: "22px",
      height: "22px",
      objectFit: "contain",
      display: "inline-block",
      verticalAlign: "middle",
    },
  });
}

function Divider() {
  return React.createElement("hr", {
    style: {
      width: "100%",
      border: 0,
      borderTop: "1px solid #e4e8f0",
      margin: "4px 0",
    },
  });
}

function Button({ props, on }) {
  const press = typeof on === "function" ? on("press") : { emit: () => {}, shouldPreventDefault: false };
  const variant = String(props?.variant || "primary").toLowerCase();
  const baseStyle = {
    width: "100%",
    minHeight: "36px",
    borderRadius: "10px",
    border: variant === "borderless" ? "none" : "1px solid #0f6adf",
    background: variant === "borderless" ? "transparent" : "#0f6adf",
    color: variant === "borderless" ? "#0f6adf" : "#ffffff",
    fontWeight: 600,
    fontSize: "13px",
    padding: variant === "borderless" ? "4px 0" : "8px 12px",
    textAlign: variant === "borderless" ? "left" : "center",
    cursor: "pointer",
  };
  return React.createElement(
    "button",
    {
      type: "button",
      style: baseStyle,
      onClick: (event) => {
        if (press.shouldPreventDefault) {
          event.preventDefault();
        }
        press.emit();
      },
    },
    asText(props?.label || "Action"),
  );
}

function Tabs({ props, children }) {
  const tabs = Array.isArray(props?.tabs) ? props.tabs : [];
  const childItems = React.Children.toArray(children);
  const [active, setActive] = useState(0);

  const tabTitles = useMemo(() => {
    if (tabs.length > 0) {
      return tabs.map((tab, idx) => asText(tab?.title || `Tab ${idx + 1}`));
    }
    return childItems.map((_item, idx) => `Tab ${idx + 1}`);
  }, [tabs, childItems]);

  const safeIndex = Math.min(Math.max(active, 0), Math.max(childItems.length - 1, 0));
  const activeChild = childItems[safeIndex] || null;

  return React.createElement(
    "div",
    {
      style: {
        display: "flex",
        flexDirection: "column",
        gap: "10px",
      },
    },
    React.createElement(
      "div",
      {
        style: {
          display: "flex",
          gap: "6px",
          flexWrap: "wrap",
        },
      },
      tabTitles.map((title, idx) =>
        React.createElement(
          "button",
          {
            key: `${title}_${idx}`,
            type: "button",
            onClick: () => setActive(idx),
            style: {
              borderRadius: "999px",
              border: idx === safeIndex ? "1px solid #0f6adf" : "1px solid #d6dbe5",
              background: idx === safeIndex ? "#ebf3ff" : "#ffffff",
              color: idx === safeIndex ? "#0f6adf" : "#334155",
              fontSize: "12px",
              fontWeight: 600,
              padding: "6px 10px",
              cursor: "pointer",
            },
          },
          title,
        ),
      ),
    ),
    activeChild,
  );
}

function Modal({ props, children }) {
  const [open, setOpen] = useState(false);
  const triggerLabel = asText(props?.trigger || "Open");
  return React.createElement(
    React.Fragment,
    null,
    React.createElement(
      "button",
      {
        type: "button",
        onClick: () => setOpen(true),
        style: {
          borderRadius: "10px",
          border: "1px solid #0f6adf",
          background: "#ebf3ff",
          color: "#0f6adf",
          fontWeight: 600,
          padding: "8px 12px",
        },
      },
      triggerLabel,
    ),
    open
      ? React.createElement(
          "div",
          {
            style: {
              position: "fixed",
              inset: 0,
              background: "rgba(17,24,39,0.45)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              padding: "16px",
              zIndex: 50,
            },
            onClick: () => setOpen(false),
          },
          React.createElement(
            "div",
            {
              style: {
                width: "min(360px, 100%)",
                borderRadius: "14px",
                border: "1px solid #d8deea",
                background: "#fff",
                padding: "14px",
                display: "flex",
                flexDirection: "column",
                gap: "10px",
              },
              onClick: (event) => event.stopPropagation(),
            },
            children,
            React.createElement(
              "button",
              {
                type: "button",
                onClick: () => setOpen(false),
                style: {
                  alignSelf: "flex-end",
                  borderRadius: "8px",
                  border: "1px solid #d0d7e4",
                  background: "#f8fafc",
                  padding: "6px 10px",
                },
              },
              "Close",
            ),
          ),
        )
      : null,
  );
}

function TextField({ props, bindings }) {
  const [value, setValue] = useBoundProp(props?.value, bindings?.value);
  return React.createElement(
    "label",
    {
      style: {
        display: "flex",
        flexDirection: "column",
        gap: "6px",
      },
    },
    React.createElement("span", { style: { fontSize: "12px", fontWeight: 600 } }, asText(props?.label || "Field")),
    React.createElement("input", {
      type: "text",
      value: asText(value),
      onChange: (event) => setValue(event.target.value),
      placeholder: asText(props?.placeholder),
      style: {
        width: "100%",
        borderRadius: "10px",
        border: "1px solid #cfd6e3",
        padding: "9px 10px",
        fontSize: "13px",
      },
    }),
  );
}

function CheckBox({ props, bindings }) {
  const [value, setValue] = useBoundProp(Boolean(props?.value), bindings?.value);
  return React.createElement(
    "label",
    {
      style: {
        display: "flex",
        alignItems: "center",
        gap: "8px",
      },
    },
    React.createElement("input", {
      type: "checkbox",
      checked: Boolean(value),
      onChange: (event) => setValue(Boolean(event.target.checked)),
    }),
    React.createElement("span", null, asText(props?.label || "Option")),
  );
}

function ChoicePicker({ props, bindings }) {
  const options = Array.isArray(props?.options) ? props.options : [];
  const [value, setValue] = useBoundProp(Array.isArray(props?.value) ? props.value : [], bindings?.value);
  const selected = new Set(Array.isArray(value) ? value.map((item) => asText(item)) : []);

  return React.createElement(
    "fieldset",
    {
      style: {
        border: "1px solid #d8deea",
        borderRadius: "10px",
        padding: "10px",
        margin: 0,
      },
    },
    React.createElement("legend", { style: { fontSize: "12px", fontWeight: 600 } }, asText(props?.label || "Choices")),
    options.map((option, idx) => {
      const optionValue = asText(option?.value || option?.label || `option_${idx}`);
      const checked = selected.has(optionValue);
      return React.createElement(
        "label",
        {
          key: `${optionValue}_${idx}`,
          style: {
            display: "flex",
            alignItems: "center",
            gap: "8px",
            marginTop: "6px",
          },
        },
        React.createElement("input", {
          type: "checkbox",
          checked,
          onChange: (event) => {
            const next = new Set(selected);
            if (event.target.checked) {
              next.add(optionValue);
            } else {
              next.delete(optionValue);
            }
            setValue(Array.from(next));
          },
        }),
        React.createElement("span", null, asText(option?.label || optionValue)),
      );
    }),
  );
}

function Slider({ props, bindings }) {
  const [value, setValue] = useBoundProp(Number(props?.value || 0), bindings?.value);
  const min = Number(props?.min || 0);
  const max = Number(props?.max || 100);
  const current = Number(value || 0);
  return React.createElement(
    "label",
    {
      style: {
        display: "flex",
        flexDirection: "column",
        gap: "6px",
      },
    },
    React.createElement("span", { style: { fontSize: "12px", fontWeight: 600 } }, `${asText(props?.label || "Value")}: ${current}`),
    React.createElement("input", {
      type: "range",
      min,
      max,
      value: Number.isFinite(current) ? current : min,
      onChange: (event) => setValue(Number(event.target.value)),
    }),
  );
}

function DateTimeInput({ props, bindings }) {
  const [value, setValue] = useBoundProp(asText(props?.value), bindings?.value);
  const enableDate = props?.enableDate !== false;
  const enableTime = Boolean(props?.enableTime);
  const inputType = enableDate && enableTime ? "datetime-local" : enableDate ? "date" : "time";
  return React.createElement(
    "label",
    {
      style: {
        display: "flex",
        flexDirection: "column",
        gap: "6px",
      },
    },
    React.createElement("span", { style: { fontSize: "12px", fontWeight: 600 } }, asText(props?.label || "Date/Time")),
    React.createElement("input", {
      type: inputType,
      value: asText(value),
      onChange: (event) => setValue(event.target.value),
      style: {
        width: "100%",
        borderRadius: "10px",
        border: "1px solid #cfd6e3",
        padding: "9px 10px",
      },
    }),
  );
}

function Video({ props }) {
  const url = asText(props?.url);
  if (!url) return null;
  return React.createElement("video", {
    src: url,
    controls: true,
    style: {
      width: "100%",
      borderRadius: "12px",
      border: "1px solid #dce1ea",
      background: "#000",
    },
  });
}

function AudioPlayer({ props }) {
  const url = asText(props?.url);
  if (!url) return null;
  return React.createElement(
    "div",
    {
      style: {
        display: "flex",
        flexDirection: "column",
        gap: "6px",
      },
    },
    props?.description
      ? React.createElement("p", { style: { margin: 0, fontSize: "12px", color: "#4b5563" } }, asText(props.description))
      : null,
    React.createElement("audio", { src: url, controls: true, style: { width: "100%" } }),
  );
}

const registry = {
  Column,
  Row,
  List,
  Card,
  Text,
  Image,
  Icon,
  Divider,
  Button,
  Tabs,
  Modal,
  TextField,
  CheckBox,
  ChoicePicker,
  Slider,
  DateTimeInput,
  Video,
  AudioPlayer,
};

const handlers = {
  openUrl: async (params) => {
    const url = asText(params?.url);
    if (url) {
      openExternal(url);
    }
  },
};

function showFallback(message) {
  if (fallbackEl) {
    fallbackEl.style.display = "block";
    fallbackEl.textContent = message;
  }
}

try {
  if (!rootEl) {
    throw new Error("Missing #root mount node.");
  }

  const spec = window.__GenUICraft_FLATSPEC__;
  if (!spec || typeof spec !== "object") {
    throw new Error("Flat-spec payload is missing or invalid.");
  }

  const appRoot = createRoot(rootEl);
  appRoot.render(
    React.createElement(
      JSONUIProvider,
      {
        registry,
        initialState: spec.state && typeof spec.state === "object" ? spec.state : {},
        handlers,
      },
      React.createElement(Renderer, {
        spec,
        registry,
      }),
    ),
  );

  window.GenUICraft_READY = true;
  window.setTimeout(() => {
    window.__GenUICraft_RENDER_DONE = true;
  }, 30);
} catch (err) {
  const detail = err && typeof err === "object" && "message" in err ? String(err.message) : String(err);
  showFallback(`Renderer error: ${detail}`);
  window.GenUICraft_READY = true;
  window.__GenUICraft_RENDER_DONE = true;
}
