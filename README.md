# Communicative Agents for Software Development

<p align="center">
  <img src='./misc/logo1.png' width=550>
</p>

All my thanks to chatdev
but as we all unknow, most simplified-chinese users are not able to use openai apis
so i made a little change to the codes, using volces（火山） apis as a demo

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
```

5.just python and enjoy it
```
python run.py --org "MyGreateCompany" --name "MyFirstProject" --task "I'd like to blablabla"(any idea you have)
```

if you have any problems, feel free to talk with me.
