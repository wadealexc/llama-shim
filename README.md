## llama-shim

`llama-shim` is a lightweight proxy that sits between llama.cpp's `llama-server` and any consumer of OpenAI-compatible APIs.

`llama-shim` wraps `llama-server` to provide a way to use model-swapping capabilities like those built into Open WebUI.

### Prerequisites

* Typescript/Node/NVM
* Have `llama-server` in your `$PATH`
* Open WebUI

### Making models visible

Move any models you want the shim to use into `llama-shim/models/`. ggufs will be pulled recursively, so it's okay to have them in subfolders.

### Attach to Open WebUI

`llama-shim` starts a server on port `8081` - if you have already been attaching OWU to `llama-server`, just switch that connection to port `8081` and you should be good to go.

### How to run

```
git clone https://github.com/wadealexc/llama-shim

cd llama-shim

npm run dev
```

#### Running via systemd

`node/nvm` are designed to run in a TTY, so typically they'll close out if you exit your SSH session. To keep this running in the background, we need to create a systemd service. Follow these steps:

##### 1. Define the service

1. `sudo nano /etc/systemd/system/llama-shim.service`

In that file, place your service definition. Below is my file; you'll need to:
- replace `fox` with your username
- ensure `WorkingDirectory` points to the repository. (if you change this, also change `ExecStart`)


```
[Unit]
# start after network is up
Description=llama-server/owu shim
After=network.target

[Service]
# Run the `run.sh` script in the project root
# This script ensures `node` and `llama-server` are in our path
# when run by systemd
User=fox
WorkingDirectory=/home/fox/llama-shim
ExecStart=/home/fox/llama-shim/run.sh

# Shutdown behavior - ensure child processes are terminated
# alongside parent
TimeoutStopSec=20
KillSignal=SIGINT
KillMode=control-group

# Restart policy
Restart=on-failure
RestartSec=10

# Send output to journal
#
# check status:
# systemctl status llama-shim
#
# view logs:
# journalctl -u llama-shim -f
StandardInput=null
#StandardOutput=append:/var/log/llama-shim.log
#StandardError=append:/var/log/llama-shim.err.log

[Install]
WantedBy=multi-user.target
```

2. Make `run.sh` (included in this repo) executable:

`chmod +x llama-shim/run.sh`

3. Ensure `run.sh` points to your `nvm` directory and includes `llama-server` in your `PATH`.

##### 2. Reload systemctl and start the service

```sh
sudo systemctl daemon-reload
sudo systemctl enable --now llama-shim
```

##### 3. View logs/status/shut down service

```sh
# check whether service is running
sudo systemctl status llama-shim

# view logs live in terminal
journalctl -u llama-shim -f -o cat

# shut down service
sudo systemctl stop llama-shim
```