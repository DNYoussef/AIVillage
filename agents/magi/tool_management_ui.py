import os
import tkinter as tk
from tkinter import ttk, messagebox
from agents.magi.tool_persistence import ToolPersistence

class ToolManagementUI:
    def __init__(self, tool_persistence: ToolPersistence):
        self.tool_persistence = tool_persistence
        self.window = tk.Tk()
        self.window.title("Tool Management")
        self.window.geometry("800x600")
        self.create_widgets()

    def create_widgets(self):
        # Tool list
        self.tool_list = ttk.Treeview(self.window, columns=("Name", "Description"), show="headings")
        self.tool_list.heading("Name", text="Name")
        self.tool_list.heading("Description", text="Description")
        self.tool_list.pack(fill=tk.BOTH, expand=True)
        self.tool_list.bind("<<TreeviewSelect>>", self.on_tool_select)

        # Tool details
        self.tool_details = tk.Text(self.window, height=10, width=80)
        self.tool_details.pack(fill=tk.BOTH, expand=True)

        # Version history
        self.version_history = ttk.Treeview(self.window, columns=("Version",), show="headings")
        self.version_history.heading("Version", text="Version")
        self.version_history.pack(fill=tk.BOTH, expand=True)
        self.version_history.bind("<<TreeviewSelect>>", self.on_version_select)

        # Buttons
        button_frame = ttk.Frame(self.window)
        button_frame.pack(fill=tk.X)

        self.new_button = ttk.Button(button_frame, text="New", command=self.new_tool)
        self.new_button.pack(side=tk.LEFT)

        self.edit_button = ttk.Button(button_frame, text="Edit", command=self.edit_tool)
        self.edit_button.pack(side=tk.LEFT)

        self.delete_button = ttk.Button(button_frame, text="Delete", command=self.delete_tool)
        self.delete_button.pack(side=tk.LEFT)

        self.refresh_button = ttk.Button(button_frame, text="Refresh", command=self.refresh_tools)
        self.refresh_button.pack(side=tk.LEFT)

        self.refresh_tools()

    def refresh_tools(self):
        self.tool_list.delete(*self.tool_list.get_children())
        tools = self.tool_persistence.load_all_tools()
        for tool_name, tool_data in tools.items():
            self.tool_list.insert("", tk.END, values=(tool_name, tool_data["description"]))

    def on_tool_select(self, event):
        selected_item = self.tool_list.focus()
        if selected_item:
            tool_name = self.tool_list.item(selected_item, "values")[0]
            tool_data = self.tool_persistence.load_tool(tool_name)
            self.tool_details.delete("1.0", tk.END)
            self.tool_details.insert(tk.END, f"Name: {tool_data['name']}\n")
            self.tool_details.insert(tk.END, f"Description: {tool_data['description']}\n")
            self.tool_details.insert(tk.END, f"Parameters: {tool_data['parameters']}\n")
            self.tool_details.insert(tk.END, f"Code:\n{tool_data['code']}")

            self.version_history.delete(*self.version_history.get_children())
            versions = self.tool_persistence.list_tool_versions(tool_name)
            for version in versions:
                self.version_history.insert("", tk.END, values=(version,))

    def on_version_select(self, event):
        selected_item = self.version_history.focus()
        if selected_item:
            version = self.version_history.item(selected_item, "values")[0]
            selected_tool = self.tool_list.focus()
            if selected_tool:
                tool_name = self.tool_list.item(selected_tool, "values")[0]
                tool_data = self.tool_persistence.load_tool_version(tool_name, version)
                self.tool_details.delete("1.0", tk.END)
                self.tool_details.insert(tk.END, f"Name: {tool_data['name']}\n")
                self.tool_details.insert(tk.END, f"Description: {tool_data['description']}\n")
                self.tool_details.insert(tk.END, f"Parameters: {tool_data['parameters']}\n")
                self.tool_details.insert(tk.END, f"Code:\n{tool_data['code']}")

    def new_tool(self):
        # Implement logic to create a new tool
        pass

    def edit_tool(self):
        selected_item = self.tool_list.focus()
        if selected_item:
            tool_name = self.tool_list.item(selected_item, "values")[0]
            # Implement logic to edit the selected tool
            pass

    def delete_tool(self):
        selected_item = self.tool_list.focus()
        if selected_item:
            tool_name = self.tool_list.item(selected_item, "values")[0]
            confirm = messagebox.askyesno("Delete Tool", f"Are you sure you want to delete the tool '{tool_name}'?")
            if confirm:
                self.tool_persistence.delete_tool(tool_name)
                self.refresh_tools()

    def run(self):
        self.window.mainloop()

if __name__ == "__main__":
    tool_persistence = ToolPersistence("tools_storage")
    ui = ToolManagementUI(tool_persistence)
    ui.run()
