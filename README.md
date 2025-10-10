## llama-shim

`llama-shim` is a lightweight proxy that sits between llama.cpp's `llama-server` and any consumer of OpenAI-compatible APIs.

`llama-shim` wraps `llama-server` to provide a way to use model-swapping capabilities like those built into Open WebUI.

### Prerequisites

* Typescript/Node/NVM
* Have `llama-server` in your `$PATH`
* Open WebUI

### How to run

```
git clone https://github.com/wadealexc/llama-shim

cd llama-shim

npm run dev
```

### Making models visible

Move any models you want the shim to use into `llama-shim/models/`. ggufs will be pulled recursively, so it's okay to have them in subfolders.

### Attach to Open WebUI

`llama-shim` starts a server on port `8081` - if you have already been attaching OWU to `llama-server`, just switch that connection to port `8081` and you should be good to go.