#!/usr/bin/env python3
"""Small operator dashboard for the spearhead visual-servo services."""

import tkinter as tk
from tkinter import messagebox, ttk

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger


SERVICES = {
    "set_start": "spearhead_servo/set_start_pose",
    "start": "spearhead_servo/start",
    "next_step": "spearhead_servo/next_step",
    "reset": "spearhead_servo/reset_to_start",
    "abort": "spearhead_servo/abort",
}


class SpearheadDashboard(Node):
    """Tk controls that invoke the visual-servo node's safe Trigger services."""

    def __init__(self, root):
        super().__init__("spearhead_dashboard")
        self.root = root
        self.service_clients = {
            name: self.create_client(Trigger, service)
            for name, service in SERVICES.items()
        }
        self.status = tk.StringVar(value="Ready. Save a start pose before the first run.")
        self._build_ui()
        self.root.after(50, self._spin_once)

    def _build_ui(self):
        self.root.title("Spearhead Control")
        self.root.resizable(False, False)
        frame = ttk.Frame(self.root, padding=16)
        frame.grid(sticky="nsew")
        ttk.Label(frame, text="Spearhead Control", font=("TkDefaultFont", 14, "bold")).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 10)
        )
        ttk.Button(
            frame,
            text="Save current position as start",
            command=lambda: self._call("set_start"),
        ).grid(row=1, column=0, columnspan=2, sticky="ew", pady=3)
        ttk.Button(frame, text="Start run", command=lambda: self._call("start")).grid(
            row=2, column=0, sticky="ew", padx=(0, 4), pady=3
        )
        ttk.Button(frame, text="Continue step", command=lambda: self._call("next_step")).grid(
            row=2, column=1, sticky="ew", padx=(4, 0), pady=3
        )
        ttk.Separator(frame).grid(row=3, column=0, columnspan=2, sticky="ew", pady=8)
        ttk.Button(
            frame,
            text="Return base to start",
            command=self._confirm_reset,
        ).grid(row=4, column=0, sticky="ew", padx=(0, 4), pady=3)
        ttk.Button(frame, text="Abort / stop", command=self._confirm_abort).grid(
            row=4, column=1, sticky="ew", padx=(4, 0), pady=3
        )
        ttk.Label(frame, textvariable=self.status, wraplength=360).grid(
            row=5, column=0, columnspan=2, sticky="w", pady=(10, 0)
        )
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)

    def _confirm_reset(self):
        if messagebox.askyesno(
            "Return base to start",
            "Stop the current sequence and drive the base back to the saved odometry start pose?",
            parent=self.root,
        ):
            self._call("reset")

    def _confirm_abort(self):
        if messagebox.askyesno(
            "Abort run",
            "Stop base motion and cancel the active spearhead sequence?",
            parent=self.root,
        ):
            self._call("abort")

    def _call(self, name):
        client = self.service_clients[name]
        if not client.service_is_ready():
            self.status.set(f"Waiting for {SERVICES[name]} ...")
            return
        self.status.set(f"Calling {SERVICES[name]} ...")
        future = client.call_async(Trigger.Request())
        future.add_done_callback(self._service_result)

    def _service_result(self, future):
        try:
            response = future.result()
            prefix = "OK" if response.success else "REJECTED"
            message = f"{prefix}: {response.message}"
        except Exception as exc:
            message = f"SERVICE ERROR: {exc}"
        self.root.after(0, self.status.set, message)

    def _spin_once(self):
        if rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.0)
            self.root.after(50, self._spin_once)


def main(args=None):
    rclpy.init(args=args)
    root = tk.Tk()
    node = SpearheadDashboard(root)
    try:
        root.mainloop()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
