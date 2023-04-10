import subprocess


class TmuxController():
    """
    Handles creating tmuxinator sessions and stopping them.
    Tmuxinator sessions are started detached.
    """

    def __init__(self, config_name: str):
        self.config_name = config_name
        self.tmux_active = False

        # Make sure that there is not already a tmux session active
        if self.is_tmux_session_running():
            print("Stopping all active tmux sessions")
            self.stop_all_tmux_sessions()

    def is_tmux_session_running(self) -> bool:
        """
        Returns whether or not there is an active tmux session.
        """
        p = subprocess.Popen(
            ["tmux", "ls"], stderr=subprocess.PIPE, stdout=subprocess.PIPE
        )
        std_out, std_err = p.communicate()
        return len(std_out) > 0

    def stop_all_tmux_sessions(self) -> bool:
        """
        Stops all active tmux sessions.
        """
        subprocess.Popen(
            ["tmux", "kill-server"], stderr=subprocess.PIPE, stdout=subprocess.PIPE
        )

    def start(self):
        """
        Start the tmux configuration.
        """
        if not self.tmux_active:
            subprocess.Popen(
                ['tmuxinator', 'start',  self.config_name, "--no-attach"],
            )
            print(f"{self.config_name} session started")

            self.tmux_active = True
        else:
            print(f"Already a session for {self.config_name}")

    def stop(self):
        """
        Stop the tmux configuration.
        """
        if self.tmux_active:
            subprocess.Popen(
                ['tmuxinator', 'stop',  self.config_name],
            )
            print(f"{self.config_name} session stopped")

            self.tmux_active = False
        else:
            print(f"{self.config_name} not started yet")
