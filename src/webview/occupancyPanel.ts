import * as vscode from "vscode";
import * as crypto from "crypto";
import type { SweepResult } from "../analyzer/occupancy_sweep";

let currentPanel: vscode.WebviewPanel | undefined;

export function showOccupancySweepPanel(
  extensionUri: vscode.Uri,
  sweep: SweepResult,
): void {
  const column = vscode.ViewColumn.Beside;

  if (currentPanel) {
    currentPanel.reveal(column);
    postSweepData(currentPanel.webview, sweep);
    return;
  }

  const panel = vscode.window.createWebviewPanel(
    "paxzas.occupancySweep",
    `Occupancy — ${sweep.gpu}`,
    column,
    {
      enableScripts: true,
      localResourceRoots: [vscode.Uri.joinPath(extensionUri, "media")],
      retainContextWhenHidden: true,
    },
  );

  currentPanel = panel;
  panel.onDidDispose(() => { currentPanel = undefined; });

  const webview = panel.webview;
  panel.webview.html = getHtml(webview, extensionUri);

  webview.onDidReceiveMessage((msg) => {
    if (msg?.type === "ready") {
      postSweepData(webview, sweep);
    }
  });
}

function postSweepData(webview: vscode.Webview, sweep: SweepResult): void {
  void webview.postMessage(sweep);
}

function getHtml(webview: vscode.Webview, extensionUri: vscode.Uri): string {
  const mediaUri = vscode.Uri.joinPath(extensionUri, "media");
  const chartJsUri = webview.asWebviewUri(
    vscode.Uri.joinPath(mediaUri, "chart.min.js"),
  );
  const htmlUri = vscode.Uri.joinPath(mediaUri, "occupancy.html");

  const fs = require("fs") as typeof import("fs");
  let html = fs.readFileSync(htmlUri.fsPath, "utf-8");

  const nonce = crypto.randomBytes(16).toString("base64");

  html = html.replace(/\{\{NONCE\}\}/g, nonce);
  html = html.replace(/\{\{CHART_JS_URI\}\}/g, chartJsUri.toString());

  return html;
}
