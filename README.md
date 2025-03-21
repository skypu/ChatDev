# Communicative Agents for Software Development

<p align="center">
  <img src='./misc/logo1.png' width=550>
</p>

All my thanks to chatdev
i made almost the least change to the codes, this version allows you to use any online-openai-style-api, besides, your local ollama is supported, too.
part of the codes are still a mass, i'd pay some time to reconstruct it(when i'm not lazy though)

# 🖥️ all you need to do is


1.Create an env for the project, as my local one:python3.9, created by conda, on my windows i just type
```
conda create --prefix A:\Pvenv\conda_p39_chatdev python=3.9
conda activate A:\Pvenv\conda_p39_chatdev
```

2.clone my git, you should just do
```
git clone https://github.com/skypu/ChatDev.git
```

3.install the requirements, you'd better cd to your chatdev dir, make sure you're in the correct conda env and type
```
pip install -r requirements.txt
```

4.make sure you have valid api-provider info, and set the enviroments, on my windows i just type
```
$env:OPENAI_API_KEY="your_openai_api_key"
$env:BASE_URL="https://ark.cn-beijing.volces.com/api/v3"
$env:DEFAULT_AI_MODEL="deepseek-r1-250120"
$env:AI_PROVIDER="openai"
```
alternatively, you could just use your local ollama with any model you've installed, all calls are free
```
$env:OPENAI_API_KEY="not needed, but keep it"
$env:BASE_URL="http://localhost:11434/api/generate"
$env:DEFAULT_AI_MODEL="gemma3:latest"
$env:AI_PROVIDER="ollama"
```

5.just python and enjoy it
```
python run.py --org "MyGreateCompany" --name "MyFirstProject" --task "I'd like to blablabla"(any idea you have)
```

## 📖 about code changes
model_backend.py:line 69; because the tiktoken module is designed for openai, deepseek models encoding methods are not included, so i set it to "cl100k_base"
to use local ollama, i modified the model_backend, added a func call_ai, a class to handel the ollama response:OllamaChatCompletion
in other files i just commented all the gpt stuffs.

if you have any problems, feel free to talk with me.

## 🤝 Acknowledgments
<a href="https://github.com/OpenBMB/ChatDev">ChatDev</a>&nbsp;&nbsp;
<a href="http://nlp.csai.tsinghua.edu.cn/"><img src="misc/thunlp.png" height=50pt></a>&nbsp;&nbsp;
<a href="https://modelbest.cn/"><img src="misc/modelbest.png" height=50pt></a>&nbsp;&nbsp;
<a href="https://github.com/OpenBMB/AgentVerse/"><img src="misc/agentverse.png" height=50pt></a>&nbsp;&nbsp;
<a href="https://github.com/OpenBMB/RepoAgent"><img src="misc/repoagent.png"  height=50pt></a>
<a href="https://app.commanddash.io/agent?github=https://github.com/OpenBMB/ChatDev"><img src="misc/CommandDash.png" height=50pt></a>

