This is just a repo to run https://huggingface.co/adamo1139/Meta_Spirit-LM-ungated

I do not own anything related to any of this.

This will probably *not* be maintained and is literally just to get you up and running with it.

This repo is messy and dumb, but it *should* work.

---

### **For some reason, the speed is much slower on Windows than Linux**

Still sorting that out, I suppose.

---

## Steps to run Windows

- `git clone` this repo
- `run.bat`

- `update.bat` will update both this repo and spiritlm

## Steps to run Linux
- `git clone` this repo
- `run.sh`

- `update.sh` will update both this repo and spiritlm

This will: 
- Set up the venv 
- Download the model 
- Download/setup spiritlm
- Get the correct version of torch 
- Then run the gradio interface.

---

- This assumes you want to use the base model, which the repo owner says works better than the expressive model overall
- This uses torch 2.5.0

---

Relevant reddit link.

https://www.reddit.com/r/LocalLLaMA/comments/1g7eotn/comment/lsq8leg/?utm_source=share&utm_medium=web2x&context=3

