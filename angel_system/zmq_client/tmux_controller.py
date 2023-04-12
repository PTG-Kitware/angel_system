import subprocess
from typing import Dict



class TmuxController():
    """
    Handles creating tmuxinator sessions and stopping them.
    Tmuxinator sessions are started detached.
    """

    def __init__(self, skill_cfg_map: Dict):
        self.tmux_active = False
        self.skill_cfg_map = skill_cfg_map

        print(f"Tmux ctrl skill:tmux-config map: {self.skill_cfg_map}")

        # Make sure that there is not already a tmux session active
        if self.is_tmux_session_running():
            print("Stopping all active tmux sessions")
            self.stop_all_tmux_sessions()

    def is_tmux_session_running(self) -> bool:
        """
        Returns whether or not there is an active tmux session. The stdout
        string from "tmux ls" is used to determine if tmux is running or not.
        If tmux is NOT running, this string is empty. If tmux is running, the
        stdout string is not empty.
        """
        p = subprocess.run(
            ["tmux", "ls"], stderr=subprocess.PIPE, stdout=subprocess.PIPE
        )
        return len(p.stdout) > 0

    def stop_all_tmux_sessions(self) -> bool:
        """
        Stops all active tmux sessions.
        """
        subprocess.run(
            ["tmux", "kill-server"], stderr=subprocess.PIPE, stdout=subprocess.PIPE
        )

    def start(self, skill_name):
        """
        Start the tmux configuration for the given skill.
        """
        if not self.tmux_active:
            # Try to get the config for this skill
            try:
                config_name = self.skill_cfg_map[skill_name]
            except KeyError:
                print(f"No config found for skill {skill_name}")

                # Stop all current tmux sessions
                self.stop_all_tmux_sessions()
                return

            subprocess.run(
                ['tmuxinator', 'start', config_name, "--no-attach"],
            )
            print(f"{config_name} session started")

            self.tmux_active = True
        else:
            print("Tmux session already running")

    def stop(self, skill_name):
        """
        Stop the tmux configuration for the given skill.
        """
        if self.tmux_active:
            # Try to get the config for this skill
            try:
                config_name = self.skill_cfg_map[skill_name]
            except KeyError:
                print(f"No config found for skill {skill_name}")

                # Stop all current tmux sessions
                self.stop_all_tmux_sessions()
                return

            subprocess.run(
                ['tmuxinator', 'stop',  config_name],
            )
            print(f"{config_name} session stopped")

            self.tmux_active = False
        else:
            print("No active tmux session.")
