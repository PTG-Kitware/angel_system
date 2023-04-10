import subprocess


class TmuxController():

    def __init__(self, config_name: str):
        self.config_name = config_name
        self.tmux_p = None
        self.tmux_active = False

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
