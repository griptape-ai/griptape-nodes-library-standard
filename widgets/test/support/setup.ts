// Stands in for the editor: shares React with the widgets, the source and the built bundles alike,
// as the editor does before it loads one, and lets tests drive React with `act`.

import * as React from "react";
import * as ReactDOM from "react-dom";
import * as ReactDOMClient from "react-dom/client";
import * as jsxRuntime from "react/jsx-runtime";
import { provideWidgetHost } from "@griptape-ai/nodes-widget-api/host";

provideWidgetHost({ react: React, reactDom: ReactDOM, reactDomClient: ReactDOMClient, jsxRuntime });
(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
