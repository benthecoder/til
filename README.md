# Today I Learned

Other TILs on the internet

- [Hashrocket - Today I Learned](https://til.hashrocket.com/)
- [Today I Learned - koaning.io](https://koaning.io/til/)
- [Simon Willison: TIL](https://til.simonwillison.net/)

## Day 1: 31 Aug 2022

- setup roadmap
- Learned about structural pattern matching in python 3.10
  - `shlex.split()` - Split the string s using shell-like syntax
  - `print(f"{var!r}`") - !r calls repr instead of str
  - cons: order of patterns have effects and wicked switching -> leads to bad design decisions, fucntion starts to lose cohesion
- watched some of lecture 3 of fsdl, learned about software testing.
  - think of test suits as a classifier, outputs pass/fail
  - trade off detection and false alarms, few missed alarms and false alarms
  - 80/20 rule for testing -> only 80% of value are from 20% of effort spent on test.
  - testing tools: pytest, doctest (for documentation), codecov
  - clean code tools: black, flake8, shellcheck (shell scripting)
  - in SWE: source code -> program with compiler
  - in ML: data -> model with training (much harder to test)

Links 🔗

- [A Closer Look At Structural Pattern Matching // New In Python 3.10!](https://www.youtube.com/watch?v=scNNi4860kk&list=PLC0nd42SBTaMpVAAHCAifm5gN2zLk2MBo&index=1)
- [shlex — Simple lexical analysis — Python 3.10.6 documentation](https://docs.python.org/3/library/shlex.html)
- [Lecture 03: Troubleshooting & Testing (FSDL 2022)](https://www.youtube.com/watch?v=RLemHNAO5Lw)

## Day 2: 1 Sep 2022

- assess multivariate normality with qq plot or Shapiro-Wilk Test
- learned about creating an R packages with devtools

Links 🔗

- [Shapiro–Wilk test - Wikipedia](https://en.wikipedia.org/wiki/Shapiro%E2%80%93Wilk_test)
- [R Packages (2e)](https://r-pkgs.org/)

## Day 3: 2 Sep 2022

- continued lecture 3
  - ML testing
    - start with testing data and checking for basic properties
    - test the training and check model is memorizing (simplest form of learning)
    - test the model itself, models are like functions, test them like functions, more advanced is to use loss values and model metrics
    - troubleshooting models
      - make it run by avoiding common errors like shape errors and OOM
      - make it fast by profiling and removing bottlenecks
      - make it right by scaling data/model
- Time series modeling in supply chain podcast
  - nested cross validation
  - intermittent time series problem
  - prediction interval
  - advice for reporting to business people: start with a punchline "my recommendation is "", this analysis shows that "" and talk about caveats and assumptions. Use the mental pyramid principle
- stats class
  - t-values = signal-to-noise ratio, used for determining statistical significance of effect size
  - SE of mean = indicates how well your sample estimate mean of population
  - [contrast](<https://en.wikipedia.org/wiki/Contrast_(statistics)>) = a linear combination of variables whos coefficients add up to zero, allowing comparison of different treatments

Links 🔗

- [google/or-tools: Google's Operations Research tools:](https://github.com/google/or-tools)
- [The Data Scientist Show - Time series modeling in supply chain, how to present like a McKinsey consultant, save the environment with data science - Sunishchal Dev - the data scientist show048](https://podcasts.google.com/feed/aHR0cHM6Ly9hbmNob3IuZm0vcy82NWQzYmI3NC9wb2RjYXN0L3Jzcw/episode/NzQ1MTAxMGYtNzNmNy00MzMxLWFjZGMtNjI1ZjVlNWVlMGI1?sa=X&ved=0CAUQkfYCahcKEwiIkq-F6ff5AhUAAAAAHQAAAAAQNQ)
- [Nested Cross-Validation for Machine Learning with Python](https://machinelearningmastery.com/nested-cross-validation-for-machine-learning-with-python/)
- [Understanding t-Tests: 1-sample, 2-sample, and Paired t-Tests](https://blog.minitab.com/en/adventures-in-statistics-2/understanding-t-tests-1-sample-2-sample-and-paired-t-tests)

## Day 4: 3 Sep 2022

- started MLOps zoomcamp
- fsdl lecture 4
  - a lot of ML is dealing with datasets
  - file system = fundemental unit of a "file", not versioned and easily overwritten, on a disk
  - object storage = API over filesystem, fundamental unit is an object, usually binary, versioning and redundancy built
  - database = mental model: everything is actually in RAM, but software makes sure everything is persisted to disk
  - data warehouse: store for OLAP vs database (OLTP)
    - ETL - data sources -> ETL -> warehouse
    - OLAP : column-oriented (mean length of comments past 30 days)
    - OLTP : row-oriented (comments for a given user)
  - data lake : unstructured data from different sources (eg logs, databases)
    - ELT dump everything in first
  - data lake + house = snowflake & databricks
  - airflow alternatives
    - prefect: <https://www.prefect.io/>
    - dagster: <https://dagster.io/>

Links 🔗

- [DataTalksClub/mlops-zoomcamp: Free MLOps course from DataTalks.Club](https://github.com/DataTalksClub/mlops-zoomcamp)
- [Lecture 04: Data Management](https://www.youtube.com/watch?v=Jlm4oqW41vY)

## Day 5: 4 Sep 2022

- started week 2 of mlops, experiment tracking with mlflow
- AB testing podcast
  - Novelty Effect : people interacting with website/service just because it's shiny and new, causing a spike in data. Solution: use a longer test duration to allow cool down from spike of interest to ensure representative sample.
  - Cannibalization: ex: releasing a discount on blenders to people who were already going to buy blenders, pulling purchases forward
  - cohort analysis
- continuing lecture 4 fsdl : data management
  - self-supervised learning
    - use parts of data to label other parts (mask sentences and predict and take patches of images and predict relationship between patches)
    - "contrasitve" training: minimize distance between image and text caption, and maximize distance bewteen image and other text
  - data augmentations
    - image: crop, skew, flip, etc. ([torchvision](https://pytorch.org/vision/stable/index.html))
      - simCLR - used augmentation to train model without labels
    - tabular: delete some cells to simulate missing data
    - text: replace words with synonym, change order ([nlpaug](https://github.com/makcedward/nlpaug))
    - audio: change speed, insert pauses, add audio effects
  - privacy ML
    - federated learning: train global model from data on local devices without requiring access to data
    - differential privacy: aggregating data to mask individual points
    - learning on encrypted data

Links 🔗

- [MLflow - A platform for the machine learning lifecycle | MLflow](https://mlflow.org/)
- [A/B testing and growth analytics at AirBnb, metrics store-Nick Handel-the data scientist show#037](https://www.youtube.com/watch?v=5AH0zToK0e4)
- [Selection Bias in Online Experimentation](https://medium.com/airbnb-engineering/selection-bias-in-online-experimentation-c3d67795cceb)
- [Glossary of A/B Testing Terms and Abbreviations](https://www.analytics-toolkit.com//glossary/)
- [google-research/simclr: SimCLRv2 - Big Self-Supervised Models are Strong Semi-Supervised Learners](https://github.com/google-research/simclr)

## Day 6: 5 Sep 2022

- Created a simple Python Package - [datastuff](https://github.com/benthecoder/datastuff)
  - [poetry](https://python-poetry.org/docs/) : manage dependency and packaging
  - [pre-commit](https://github.com/pre-commit/pre-commit): ensure proper code formatting before commits
  - [scriv](https://github.com/nedbat/scriv) : changelog management and generation
  - [tox](https://tox.wiki/en/latest/) : automatic testing and linting
  - codecov: get coverage reports
  - github actions: CI/CD for linting and tresting and publishing to PyPI
- started lecture 5 fsdl : deployment
  - start with prototyping : streamlit / gradio
  - architectures
    - model in service : package model within your web server, UI + model loaded together
    - model in database : preprocess -> load -> batch prediction -> store in database (model is run offline)
    - model as service : run model on its own web server, backend interact with model by sending requests and receiving responses

Links 🔗

- [How to create a Python package in 2022 | Mathspp](https://mathspp.com/blog/how-to-create-a-python-package-in-2022)
- [Lecture 05: Deployment (FSDL 2022)](https://www.youtube.com/watch?v=W3hKjXg7fXM)

## Day 7: 6 Sep 2022

- Watched Lecture 1 of FSDL 2021
  - universal function approximation theoreom
    - def: give any cont. f(x), if a 2-layer NN has enough hidden units, there is a choice of Ws that allow it to approx. f(x)
    - in other words, extremenly large neural networks can represent anything
  - conditioning NNs
    - weight initialization
    - normalization
    - second order methods (computationally expensive): newton's method, natural gradient
      - adagrad, Adam approximates second order
    - automatic differentiation
      - only need to program forward function f(x, w), PyTorch automatically computes all derivatives.
  - NN architecture considerations
    - data efficiency
      - encode prior knowledge into architecture
        - CV: CNNs = spatial translation invariance (CNNs simulate how our eye works!)
        - NLP: RNN = temporal invariance (rules of language in any position of the sequence)
    - optimization landscape / conditioning
      - depth over width, skip connections, normalization
    - computational / parameter efficiency
  - CUDA = NVIDIA’s language/API for programming on the graphics card

Links 🔗

- [Lecture 1: Deep Learning Fundamentals FSDL 2021](https://www.youtube.com/watch?v=fGxWfEuUu0w&list=PL1T8fO7ArWlcWg04OgNiJy91PywMKT2lv&index=1)
- [A visual proof that neural nets can compute any function](http://neuralnetworksanddeeplearning.com/chap4.html)

## Day 8: 7 Sep 2022

- Event logging @ airbnb
  - logs informs decision-making in business and drives product development, ex: logging events are a major source for training ML models for search ranking of listings
  - service-level agreements (SLAs) = define and measure the level of service a given vendor, product, or internal team will deliver—as well as potential remedies if they fail to deliver

Links 🔗

- [The Log: What every software engineer should know about real-time data's unifying abstraction | LinkedIn Engineering](https://engineering.linkedin.com/distributed-systems/log-what-every-software-engineer-should-know-about-real-time-datas-unifying)
- [Scaling Spark Streaming for Logging Event Ingestion | by Hao Wang | The Airbnb Tech Blog | Medium](https://medium.com/airbnb-engineering/scaling-spark-streaming-for-logging-event-ingestion-4a03141d135d)
- [CS 230 - Recurrent Neural Networks Cheatsheet](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-recurrent-neural-networks)

## Day 9: 8 Sep 2022

- Feature stores
  - a data management layer for machine learning that allows to share & discover features and create more effective machine learning pipelines.
  - they make it easy to
    - automate feature computation, backfills, logging
    - share and reuse feature pipelines across teams
    - track feature version, lineage, metadata
    - monitoring health of feature pipeline
  - main components
    - Serving: serve with high-performance API backed by low-latency database
    - Storage:
      - online : model training (months - years of data in data warehouse/lakes like S3, BigQuery)
      - offline : model serving/inference (lastest data that models the current state of the world in key-value stores like Redis, Cassandra)
    - Transformation
      - batch transform : data warehouse/lakes (ex: product category)
      - streaming transform : kafka, kinesis (ex: # of clicks in last 30 min)
      - on-demand transform : application (ex: similarity score between listing and search query or user ip) \*data available at the time of prediction
    - Monitoring
      - track data quality for drift and training-serving skew
      - monitor operational metrics (capacity on storage and throughput and latency on serving)
    - Registry : single source of truth for info about a feature
      - explore, develop, collaborate, and publish new definitions
      - schedule and configure data ingestion, transformation and storage
      - metadata for feature definition (ownership, lineage tracking)

Links 🔗

- [Feature Store For ML](https://www.featurestore.org/)
- [What is a Feature Store? | Tecton](https://www.tecton.ai/blog/what-is-a-feature-store/)

## Day 10: 9 Sep 2022

- lecture 5 fsdl continued
  - rest APIs : serve predictions in response to HTTP request
    - grpc (used in tf serving) and graphql are alternatives
  - dependency management
    - constraint dependency of model : [ONNX](https://onnx.ai/) - define network in any language and run anywhere
    - container : docker (VM without OS with docker engine) or use ([Cog](https://github.com/replicate/cog), [BentoML](https://www.bentoml.com/))
  - performance optimization
    - GPU? = Pros: same hardware for training and increase throughput, Cons: complex to setup
    - concurrency = multiple copies of model on different CPUs/cores (careful thread tuning)
    - distillation = tran smaller model to imitate larger one ([distilbert](https://huggingface.co/docs/transformers/model_doc/distilbert))
    - quantization = execute all operation of model with smaller numerical repr (INT8), tradeoff in accuracy
    - caching = some inputs are more common than others (basic: use functools.cache)
    - batching = gather inputs, run prediction, split and return to individual users
    - share GPU = run multiple models on the same GPU
    - model serving libraries = [Ray Serve](https://docs.ray.io/en/latest/serve/index.html)
  - horizontal scaling = split traffic across multiple machine that has a copy of model
    - container orchestration : kubernetes manages docker containers and run across machines [Kubeflow](https://www.kubeflow.org/)
    - serverless : package model and dependency into .zip or docker container with a single entry point (`model.predict()`) -> deploy to AWS lambda
  - model rollout : how you manage and update model services
    - roll out gradually : incrementally test new model
    - roll back instantly : if something's wrong, pull back model instantly
    - split traffic between versions : test differences between old and new (A/B test)

Links 🔗

- [Docker overview](https://docs.docker.com/get-started/overview/#docker-architecture)
- [How We Scaled Bert To Serve 1+ Billion Daily Requests on CPU](https://www.youtube.com/watch?v=Nw77sEAn_Js)
- [Research Guide: Model Distillation Techniques for Deep Learning](https://heartbeat.comet.ml/research-guide-model-distillation-techniques-for-deep-learning-4a100801c0eb)
- [Introduction to Quantization on PyTorch](https://pytorch.org/blog/introduction-to-quantization-on-pytorch/)
- [🤗 Optimum](https://huggingface.co/docs/optimum/index)
- [functools — Higher-order functions and operations on callable objects — Python 3.10.7 documentation](https://docs.python.org/3/library/functools.html)

## Day 11: 10 Sep 2022

- Multi-armed bandit
  - Goal: determine best or most profitable outcome through a series of choices

Links 🔗

- [When Life Gives You Lemons | TigYog](https://tigyog.app/d/L:X07z8laLyz/r/when-life-gives-you-lemons)
- [The Multi-Armed Bandit Problem and Its Solutions | Lil'Log](https://lilianweng.github.io/posts/2018-01-23-multi-armed-bandit/)
- [Multi-armed bandit - Optimizely](https://www.optimizely.com/optimization-glossary/multi-armed-bandit/)

## Day 12: 11 Sep 2022

- watched refactoring data science project
  - protocol vs ABC
  - runner class for train vs test
  - function composition (PyTorch: Seqeuential())
  - information expert principle: the way data flows informs the design

Links 🔗

- [Refactoring A Data Science Project Part 1 - Abstraction and Composition](https://www.youtube.com/watch?v=ka70COItN40)
- [Python Type Hints - Duck typing with Protocol - Adam Johnson](https://adamj.eu/tech/2021/05/18/python-type-hints-duck-typing-with-protocol/)
- [Sequential — PyTorch 1.12 documentation](https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html)
- [Every Data Scientist Should Know This Design Principle](https://aigents.co/data-science-blog/publication/every-data-scientist-should-know-this-design-principle)

## Day 13: 12 Sep 2022

- starting working on data science interview questions
- recapped course 1 - supervised machine learning

Links 🔗

- [Supervised Machine Learning: Regression and Classification | Coursera](https://www.coursera.org/learn/machine-learning)

## Day 14: 13 Sep 2022

- split-plot design
- did more interview questions by interviewqs

Links 🔗

- [14.3 - The Split-Plot Designs | STAT 503](https://online.stat.psu.edu/stat503/lesson/14/14.3)

## Day 15: 14 Sep 2022

- 2 more interview questions by interviewqs
- read through kaggle notebooks for nlp and time series

Links 🔗

- [NLP with Disaster Tweets - EDA, Cleaning and BERT | Kaggle](https://www.kaggle.com/code/gunesevitan/nlp-with-disaster-tweets-eda-cleaning-and-bert#0.-Introduction-and-References)

## Day 16: 15 Sep 2022

- sql questions on datalemur
- advanced learning algorithms ML specialization Andrew NG week 1

Links 🔗

- [SQL Server EXISTS and NOT EXISTS](https://blog.devart.com/sql-exists-and-not-exists.html)

## Day 17: 16 Sep 2022

- finished week 1 of advanced learning algorithms ML specialization
  - learned about sequential, activation functions, epochs, batches, logistic unit, vectorization, numpy broadcasting

Links 🔗

- [Broadcasting — NumPy v1.23 Manual](https://numpy.org/doc/stable/user/basics.broadcasting.html)

## Day 18: 17 Sep 2022

- week 2 of advanced learning algorithms ML specialization
  - activation functions
    - why relu > sigmoid?
      - faster to compute
      - learns faster
  - softmax
    - multiclass classification, generalizes loss function into -log a_i for multiple classes

Links 🔗

- [machine learning - What are the advantages of ReLU over sigmoid function in deep neural networks? - Cross Validated](https://stats.stackexchange.com/questions/126238/what-are-the-advantages-of-relu-over-sigmoid-function-in-deep-neural-networks)

## Day 19: 18 Sep 2022

- continual learning
  - most valuable signals to monitor by importance
    - outcomes and feedback by user (clicks, churn, flags, likes, interactions, etc.)
    - model performance metrics (offline metrics: accuracy)
    - proxy metrics (correlated with bad model performance i.e. repetitive outputs, fewer personalized responses)
    - data quality (accuracy, completeness, consistency, timeliness, et.)
    - distribution drift
    - system metrics

Links 🔗

- [Lecture 06: Continual Learning (FSDL 2022)](https://www.youtube.com/watch?v=nra0Tt3a-Oc)

## Day 20: 19 Sep 2022

- foundational models
  - transfer learning : fast training on little data
    - lot of data -> large model (costly)
    - much less data -> pretrained model + new layers
  - word2vec learns embedding that maximises cosine similarity of words that frequently co-occur
  - transformer components
    - self-attention
    - positional encoding
    - layer normalization

Links 🔗

- [Lecture 07: Foundation Models (FSDL 2022)](https://www.youtube.com/watch?v=Rm11UeGwGgk&t=1087s)

## Day 21: 20 Sep 2022

- foundation models
  - generative pretrained transformer (GPT/GPT-2) is decoder only (masked self-attention) which means you can only look at points before your input
  - BERT is encoder only (no attention masking)
  - T5: text-to-text transfer transformer (input and output are text strings)
  - GPT-3 (175B params)
    - zero-shot : (t5 model) give description of task and get prediction
    - one-shot : give one example and input
    - few-shot : give few examples and input
    - the more shots & larget the model -> the better performance (still increasing)
  - instruct-GPT
    - had humans rank GPT-3 output and use RL to fine-tine model
  - retrieval-enhanced transformer (RETRO)
    - learn langauge in params, and retrieve facts from large database
    - BERT-encode sentences, store in large DB, then fetch matching sentences and put into prompt.
  - Chinchilla (trained over 400 LM's from 16 to 500B)
    - found that most LLMs are undertrained
    - PAY ATTENTION TO DATASETS

Links 🔗

- [GPT-2: 1.5B Release](https://openai.com/blog/gpt-2-1-5b-release/)
- [Google AI Blog: Open Sourcing BERT: State-of-the-Art Pre-training for Natural Language Processing](https://ai.googleblog.com/2018/11/open-sourcing-bert-state-of-art-pre.html)
- [Google AI Blog: Exploring Transfer Learning with T5: the Text-To-Text Transfer Transformer](https://ai.googleblog.com/2020/02/exploring-transfer-learning-with-t5.html)
- [Aligning Language Models to Follow Instructions](https://openai.com/blog/instruction-following/)
- [Improving language models by retrieving from trillions of tokens](https://www.deepmind.com/publications/improving-language-models-by-retrieving-from-trillions-of-tokens)

## Day 22: 21 Sep 2022

- semantic search
  - text + query both embedded with LLMs -> compute cosine similarity between both vectors
  - challenge is computation, libraries like FAISS and ScaNN makes it feasible
  - open source: DeepSet Haystack, deepset, jina
- transformers
  - positional encoding (allows NN learn order of sequence)
  - self-attention
    - attention: gives text model mechanism to "look at" every signle word in input sentence when making decision about prediction
    - self-attention: gives model ability to automatically build up meaningful underlying meaning and patttern in langauge. ex: it can understand a word in the context of words around it (crashed the server vs server in restaurant)

Links 🔗

- [Transformers, Explained: Understand the Model Behind GPT-3, BERT, and T5](https://daleonai.com/transformers-explained)
- [[1409.0473] Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
- [Illustrated Guide to Transformers Neural Network: A step by step explanation](https://www.youtube.com/watch?v=4Bdc55j80l8&t=207s)

## Day 23: 22 Sep 2022

- word segmentation
  - tokenization is a general segmentation problem, what if text is a stream of characters, how do you know where to split into sentence?
  - Chinese word segmentation is a hard problem as there is no visual representation of word boundaries. 爱国人 can be 爱国 / 人 or 爱 / 国人
- normalization
  - stemming and lemmatization both convert an inflected word form to a canonical word form
  - stemmers : a quick and lightweight way to reduce vocab size but sacrifices information (popular for info retrieval)
  - lemmetization: convert to meaningful base form, looks at context of word. (good choice for compiling vocab)
  - "Another normalization task involves identifying non-standard words, including numbers, abbreviations, and dates, and mapping any such tokens to a special vocabulary. For example, every decimal number could be mapped to a single token 0.0, and every acronym could be mapped to AAA. This keeps the vocabulary small and improves the accuracy of many language modeling tasks"

Links 🔗

- [Text segmentation - Wikipedia](https://en.wikipedia.org/wiki/Text_segmentation)

## Day 24: 23 Sep 2022

- started learning scala

Links 🔗

- [benthecoder/scala: learning scala](https://github.com/benthecoder/scala)

## Day 25: 24 Sep 2022

- CLIP (Contrastive Language–Image Pre-training)
  - transfoer to encode text, resnet/visual transfomer to encode image
  - does contrastive training (maximize cosine similarity of correct image-text pair)
  - inference?
    - zero-shot : encode new label, see which embedding is closest to image embedding
  - image -> embedding and text -> embedding, not directly image -> text and vice versa
  - you can search image by text and vice versa since CLIP embeds image and text into a shared space
  - image generation
- unCLIP (DALL-E 2)
  - CLIP : text + image encoder
  - prior : map text embedding to image embedding
    - why do you need prior? there are infinitely many text description that matches a single image
  - decoder : map image embedding to image
- Diffusion models
  - X_T (pure noise) <-> X_t <-> X_t-1 <-> X_0 (original vector)
  - X_T is pure noise, X_0 is original image
  - add noise to original vector going from X_t-1 to X_t
  - train model to denoise vector, with information on timestep [X_t, t] to X_t-1
  - can generate infinite training data with different types of noise
  - trained model can go from pure noise to original vector (or interpolation)
  - add additional features e.g. embeddings, captions, labels
  - entire sequence
    - [encode text, clip txt embedding, diffusion time steps, noised image embedding] -> de-noised image embedding
  - unCLIP decoder
    - classic U-net
    - basic idea is diffusion model is trained to go from image of random noise to progressively higher-res images
  - stable diffusion
    - like CLIP but diffuse in lower-dim latent space, and then decode back into pixel space
    - trick to work on smaller dataset

Links 🔗

- [CLIP: Connecting Text and Images](https://openai.com/blog/clip/)
- [How DALL-E 2 Actually Works](https://www.assemblyai.com/blog/how-dall-e-2-actually-works/)
- [An overview of Unet architectures for semantic segmentation and biomedical image segmentation | AI Summer](https://theaisummer.com/unet-architectures/)
- [What are Diffusion Models? | Lil'Log](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)

## Day 26: 25 Sep 2022

- Build the future Sam Altman
  - how to pick what's important to work on?
    - choose intersection of what you're good at, what you enjoy, and what way you can create value fo the world
  - how to get things done
    - three things: focus, personal connection and self-belief
  - advice for twenties
    - work hard and get better every day, leverage compound effect
    - "work harder than most people think you should"
  - when to give up?
    - base on an internal, not external (critics), decision, when you run out of ideas and something isn't working
  - taking risks
    - risk is not doing something that you will spend the rest of your life regretting
    - history belongs to the doers
  - "when you're thinking about a startup, it's really worthwhile to think about something you're willing to make a very long-term commitment to because that is where the current void in the market is"
  - the most successful people have strong opinions strongly held about the future
- What is Kafka?
  - capable of ingesting and processing trillions of records per day without any perceptible performance lag as volumes scale
  - 3 primary capabilities
    - enables applications to publish or subscribe to data or event streams
    - stores records accurately (sequential order) in a fault-tolerant way
    - processes records in real-time
  - 4 APIs
    - Producer API: allows apps to make streams of data, creates records and produces to topics (an ordered list of events that persists temporarily depending on storage)
    - Consumer API: subscribes to one or more topics and listens and ingests data
    - Streams API: Producer -> transform data (Streams) -> Consumer. In a simple application, Producer -> Consumer is enough where data doesn't change.
    - Connector API: Reusable producer and consumer (package code for other developers to reuse and integrate for their purposes)
- Watched a bit of lecture 1 of Stanford CS224N NLP

Links 🔗

- [Sam Altman : How to Build the Future](https://www.youtube.com/watch?v=sYMqVwsewSg)
- [Apache Kafka - IBM](https://www.ibm.com/cloud/learn/apache-kafka)
- [What is Kafka?](https://www.youtube.com/watch?v=aj9CDZm0Glc)
- [Stanford CS224N: NLP with Deep Learning | Winter 2021 | Lecture 1 - Intro & Word Vectors](https://www.youtube.com/watch?v=rmVRLeJRkl4&list=PLoROMvodv4rOSH4v6133s9LFPRHjEmbmJ)

## Day 27: 26 Sep 2022

- Experimentation : A way for companies to understand if product efforts are working
  - process: build a small version, launch to a small percentage of users, measure impact, learn and try again
  - example: impact of a new price comparison tool at Amazon. launch a basic version of the tool, measure purchase behaviour between groups, make decision
  - statistics of sampling requires a meaningfully sized base of users for proper statistically significant experiments
  - experimentation logsitics
    - feature flagging and randomization
      - completely random sampling of small percentage of users who gets the feature, also keep track of who sees and doesn't see to tie that back to user behaviour and impact of experiment
    - measuring metrics
      - amazon: purchase volumne, other experiments: time on page, conversion rate, etc.
    - run time and statistical significance
      - "waiting to reach statistical significance" = waiting until sample size is large enough to be representative of whole population
    - managing experimental volume
      - multiple experiments by different teams require paying attention to which users are in which experiments, ideally one user should see only one experiment.
- Bayesian Statistics
  - Frequentist inference : parameters in model are unknown but fixed
  - Bayesian statistics: unknown and uncertain
    - we put prior on them to express uncertainty about them.
    - it makes doing inference flexible, you can trasnform parameters and transform them back later, and we don't have to make as strong distributional assumptions
  - what is a counterfactual?
    - example: effect of masking policy. Factual world where Covid happened where some schools are masked and some arent. Turn back time, and change mask policy and observe effect. That's the counterfactual world.Essentially have two worlds where everything else is constant (the same kids, same activities, same community, etc.) except for the mask policy.
    - To create this counterfactual world -> randomization
  - "The important part is deeply engaging with the subject matter and researchers in the field and understanding the assumptions and communicating it better"

Links 🔗

- [[Technically dispatch] what is A/B testing and what did LinkedIn do wrong](https://technically.substack.com/p/technically-dispatch-what-is-ab-testing)
- [Experiments at Airbnb](https://medium.com/airbnb-engineering/experiments-at-airbnb-e2db3abf39e7)
- [Under the Hood of Uber's Experimentation Platform | Uber Blog](https://www.uber.com/blog/xp/)
- [Challenges in Experimentation | Lyft Engineering](https://eng.lyft.com/challenges-in-experimentation-be9ab98a7ef4)
- [Experimentation @ AIRBNB](https://medium.com/@benedictxneo/list/experimentation-40c2327e8667)
- [SDS 607: Inferring Causality - SuperDataScience](https://www.superdatascience.com/podcast/inferring-causality)
- [Causal Inference 3: Counterfactuals](https://www.inference.vc/causal-inference-3-counterfactuals/)

## Day 28: 27 Sep 2022

- check no of rows fast
  - `s = !wc -l {TRAIN_PATH}`
  - `n_rows = int(s[0].split(' ')[0])+1`
- anomaly detection
  - anomalies = certain patterns (or incidents) in a set of data samples that do not conform to an agreed-upon notion of normal behavior in a given context
  - two approaches
    - 1\. rule-based
      - uses a set of rules which rely on knowledge of domain experts
      - experts specify characteristics of rule-based functions to discover anomalies
      - bad: expensive, time-consuming at scale, requires constant supervision to stay up-to-date, biased, and suitable for real-time analysis
    - 2\. model-based
      - scalable and suited for real time analysis
      - highly rely on availablity (often labeled) context-specific data
      - three kinds
        - supervised : use labeled data to distinguish between anomalous and benign
        - semi-supervised : small set of benign + large amount of unlabeled data, learns distribution of benign samples and leverage that knowledge for identifying anomalous samples at inference time.
        - unsupervised : do not require any labeled data samples
  - multi-label task
    - accuracy, precision, recall, f0.5, f1, and f2 scores (Fbeta measures), exact match ratio (EMR) score, Hamming loss, and Hamming score

Links 🔗

- [How to import a CSV file of 55 million rows | Kaggle](https://www.kaggle.com/code/szelee/how-to-import-a-csv-file-of-55-million-rows/notebook)
- [Reducing DataFrame memory size by ~65% | Kaggle](https://www.kaggle.com/code/arjanso/reducing-dataframe-memory-size-by-65/notebook)
- [Machine Learning for Fraud Detection in Streaming Services](https://netflixtechblog.medium.com/machine-learning-for-fraud-detection-in-streaming-services-b0b4ef3be3f6)
- [[2203.02124] Abuse and Fraud Detection in Streaming Services Using Heuristic-Aware Machine Learning](https://arxiv.org/abs/2203.02124)
- [A Gentle Introduction to the Fbeta-Measure for Machine Learning](https://machinelearningmastery.com/fbeta-measure-for-machine-learning/)
- [Metrics for Multilabel Classification | Mustafa Murat ARAT](https://mmuratarat.github.io/2020-01-25/multilabel_classification_metrics)

## Day 29: 28 Sep 2022

- Testing in Python
  - Levels of testing
    - unit testing : test independent modules and functions
    - integration testing : test how functions interact with each other
    - system testing : testing the entire system
    - acceptance testing: test whether system's operation aligns with business requirement
  - Mocking
    - replace object in plac eof a dependency that has certain expectations
    - example: testing a function that sends an email. you don't want to send an email everytime you run the test, so you mock the email sending function and check if it was called with the right arguments
  - fixtures
    - a particular environment that must be set up before a test is run
    - example: a test that requires a database connection. you can create a fixture that creates a database connection and returns it, and then use that fixture in the test
- 2 SQL questions on data lemur

Links 🔗

- [Testing practices for data science applications using Python](https://medium.com/data-science-at-microsoft/testing-practices-for-data-science-applications-using-python-71c271cd8b5e)
- [PYTEST DS example](https://github.com/Jazz4299/PYTEST)
- [pytest: How to mock in Python](https://changhsinlee.com/pytest-mock/)
- [pytest fixtures: explicit, modular, scalable — pytest documentation](https://docs.pytest.org/en/6.2.x/fixture.html#what-fixtures-are)
- [abhivaikar/howtheytest: A collection of public resources about how software companies test their software](https://github.com/abhivaikar/howtheytest)

## Day 30: 29 Sep 2022

- How to lead
  - there is not a single archetype
  - 3 qualities
    - great communicator : clear, concise communication
      - advice: jot down your thoughts and think how to express it in clearer and clearer ways
    - good judgement on people : be able to identify and hire the right people
      - advice: meet a lot of people and hone your instincts
    - strong personal integrity and commitment
      - integrity = standing for something meaningful beyond themselves and being motivated by things outside of their narrow personal interest, avoid behaviour that diminishes trust and credibility like favoritism, conflict of interest
      - Commitment = making your work into a life mission, people see this and respect it and follow it
      - advice: hold yourself accountable to the transparency test
  - success metric for leadership is trust
    - art of trust : show empathy and good judgement, having good timing in confronting issues
    - science of trust : be right about empirical questions in your business

Links 🔗

- [How to Lead : YC Startup Library | Y Combinator](https://www.ycombinator.com/library/6s-how-to-lead)

## Day 31: 30 Sep 2022

- MLops MLFlow lectures
- to make `.gitignore` work for files that have been commited, do `git rm -rf --cached .`
- creating profiles with aws cli : `aws configure --profile {profile_name}`

Links 🔗

- [MLOps Zoomcamp 2.6 - MLflow in practice](https://www.youtube.com/watch?v=1ykg4YmbFVA&list=PL3MmuxUbc_hIUISrluw_A7wDSmfOhErJK&index=17)

## Day 32: 1 Oct 2022

- MLops Prefect lectures

Links 🔗

- [Prefect - The New Standard in Dataflow Automation - Prefect](https://www.prefect.io/)
- [Why Not Airflow? | Prefect Docs](https://docs-v1.prefect.io/core/about_prefect/why-not-airflow.html#the-scheduler-service)

## Day 33: 2 Oct 2022

- mlops deployment lectures
  - 3 ways of deploying models
    - batch offline
    - web services
    - streaming
- what is dbt
  - a tool for data transformation in data warehouse
  - it's more than organizing .sql files and running them for you + managing DDL:
    - Built in testing
    - Built in dependency management for your models
    - Focus on version control
    - CI
    - Orchestration with dbt Cloud
    - Jinja, which enables some pretty neat functionality

Links 🔗

- [Learn Analytics Engineering with dbt](https://courses.getdbt.com/collections)

## Day 34: 3 Oct 2022

- Michael Seibel's Learnings
  - successful seed round != product-market fit
  - Don't take on too many problems / products
  - Understand business model (example: pricing for SMB, but selling to Enterprise)
  - Understand when it's right time to sell
  - Don't assume investors will be LARGE differentiator
  - Establish best practices around hiring (i.e. intelligent interview process, clear mission/culture, roles/responsibilities)
  - Establish best practices around management
  - Create transparency around how the business is doing (business KPIs and product KPIs)
  - Clearly defining roles/responsibilities between the founders
  - Have level three (intense, but pragmatic) conversations to alleviate tensions between founders
  - Don't assume Series A will be as easy to raise as seed/angel rounds

Links 🔗

- [A Decade of Learnings from Y Combinator's CEO Michael Seibel](https://www.youtube.com/watch?v=0MGNf1BIuxA)

## Day 35: 4 Oct 2022

- Linux commands
  - `nohup tar -zxvf {tar_file} &` to expand tar files
    - `nohup` - makes program ignore HUP signal, allowing it to run after user logs out
    - `&` - related to shell job control, allowing users to continue work in curreent shell session
  - `top` - shows running processes
  - `ps -ef | grep {process_name}` - grab specific process
  - `kill -9 {pid}` to kill process

Links 🔗

- [Data Science at the Command Line, 2e](https://datascienceatthecommandline.com/2e/index.html)

## Day 36: 5 Oct 2022

- finished all easy SQL questions on data lemur
- learnings
  - subquery vs CTE
    - subquery is not reusable.
    - CTE is reusable, easier to read and maintainable and can be used recursively
  - `=` vs `LIKE`
    - = is used for exact match
    - LIKE matches strings character by character
  - postgrtesql findings
    - `DATE_PART('year', {date_column}::DATE)` - returns year of date
    - `ROUND(division_res::DECIMAL, 2)` ROUND requires type to be numeric, division is type double, cast to DECIMAL for it to work
    - `GROUP BY 1` - group by first column
    - ways to count events
      - `COUNT(CASE WHEN event_type = 'click' THEN 1 ELSE NULL END)`
      - `SUM(CASE WHEN event_type = 'click' THEN 1 ELSE 0 END)`
      - `SUM(1) FILTER (WHERE event_type = 'click')`
    - check if action is 1 day after sign up with `INTERVAL`
      - `texts.action_date = emails.signup_date + INTERVAL '1 day'`
  - why do we need window functions?
    - definition: to operate on a subset of rows in a table
    - basic operations in SQL apply to whole datasets
      - where - filters whole table
      - functions - applies to entire column
      - group by - every aggregate field will be grouped by every non-aggregate field
    - if you want to operate in two different ways, you can either make CTEs or subqueries and do a self join, or you can use window functions

Links 🔗

- [sql - Difference between CTE and SubQuery? - Stack Overflow](https://stackoverflow.com/questions/706972/difference-between-cte-and-subquery)
- [CTEs versus Subqueries · Alisa in Techland](https://www.alisa-in.tech/post/2019-10-02-ctes/)
- [SQL Window Functions: I guess that's why they call it window pain · Alisa in Techland](https://www.alisa-in.tech/post/2021-02-19-window/)

## Day 37: 6 Oct 2022

- ethics of ML
  - is the model fair
    - fairness is possible but requires trade-offs
    - example -> COMPAS : predicting rearrest probability and to be less biased, calibrated for all racial groups
      - more false positive for black and more false negative for white
      - there is no way to make it fair for all groups, equalizing FPR and PPV across groups would be unfair to white people instead
      - always ask, "Should this model be built at all?"
    - representation in ML - model cards that explains what the model can and cannot do
  - is the system accountable
    - accountability is easier than interpretability
    - explanation and accountability are human rights
    - off-the-shelf methods for introspecting DNNs are not robust
  - who owns the data?
    - large datasets are crawled from the internet
    - data governance
    - dataset cards
    - data ethics checklist
  - medicine leads the way for responsible ML
    - ML did not work well for COVID due to bad practices and lack of sufficient external validation and testing.
    - medicine has strong culture of ethics that equips it to handle serious problems
      - Hippocratic oath : do no harm
      - tech : move fast and break things (BAD FOR MEDICINE)
    - clinical trials standards -> ML SPIRIT-AI and CONSORT-AI initiative
    - progress is being made, but error analysis (Adverse Event Analysis), data handling and code access is worst compliance
    - "medical algorithmic audit" for medical ML
      - adversarial testing : coming up with different inputs into model, behavioural check

Links 🔗

- [Lecture 09: Ethics (FSDL 2022)](https://www.youtube.com/watch?v=7FQpbYTqjAA&list=PL1T8fO7ArWleMMI8KPJ_5D5XSlovTW_Ur)
- [Attack discrimination with smarter machine learning](https://research.google.com/bigpicture/attacking-discrimination-in-ml/)
- Reverse Engineer DNN - [Transformer Circuits Thread](https://transformer-circuits.pub/)
- [Machine Learning CO2 Impact Calculator](https://mlco2.github.io/impact/#compute)
- [CodeCarbon.io](https://codecarbon.io/)
- [Have I Been Trained?](https://haveibeentrained.com/)
- [Model Cards](https://huggingface.co/docs/hub/model-cards)
- [ethics checklist for data science projects](https://deon.drivendata.org/examples/)

## Day 38: 7 Oct 2022

- statistics class
  - one and two way anova
    - one way : how one factor affects the response
    - two way: how two factors affect the response
  - doing analysis
    - start with anova table
    - use f test based on type III to answer "standard" factorial questions
      - what are means, differences/contrasts that answer important questions
      - how precise are means, differences, contrasts
      - most useful number in ANOVA table is often MSE
      - plot the means in a way that communicates the key results
    - useful things to check
      - check residuals (important if effects have large SEs)
        - look for equal variance, additive effects
        - if not reasonable correcting often increases power
      - if balanced data
        - check if Type I SS == Type III SS (should be same when balanced)
      - check d.f for highest order interaction
        - should be product of main effect d.f.
        - if not, missing data
- hands on ml chapter 4
  - normal equation
    - closed-form solution (mathematical equation that gives the result directly) to finding weights that minimizes cost function
  - pseudoinverse (uses SVD)
    - more efficient that Normal equation and hadnle edge case (m < n) can still define a pseudoinverse whereas X.T \* X is not invertible
  - gradient descent
    - batch (better name : full gradient descent)
      - trains on full training set but still faster than normal equation
      - convergance rate is $O(1/\epsilon)$ iterations with fixed learning rate
    - stochastic
      - pickes random instance in training set at every step and computes gradients based only on that single instance
      - possible to train on huge training set as only one instance needs to be in memory at each iteration (can be implemented as out-of-core algorithm)
      - stochastic (random) nature is good for jumping out of local minima but bad for settling at minimum, so we implement a learning schedule that gradually reduces the learning rate
      - training must be IID to ensure parameters get pulled toward global optimum, shuffle instances during training or at beginning of each epoch
    - mini-batch
      - computes gradient on small random set of instances called mini-batches
      - main advantage over stochastic : performance boost from hardware optimization of matrix operations
  - regularized linear regression
    - ridge
      - uses l2 norm
      - sets weights close to zero
    - lasso
      - uses l1 norm
      - sets weight to zero -> automatically performs feature selection and outputs a sparse model
    - elastic net
      - middle ground between ridge and lasso
      - uses mix ratio r to controll mix of l1 and l2 (r = 0 -> ridge, r = 1 -> lasso)
    - advice
      - always have a bit of regularization
      - scale the data before regularization
      - ridege is good default if you want to keep all features
      - use lasso to reduce features but elastic net if lasso is erratic when m > n or several features are strongly correlated

Links 🔗

- [Anova – Type I/II/III SS explained](https://md.psych.bio.uni-goettingen.de/mv/unit/lm_cat/lm_cat_unbal_ss_explained.html#:~:text=The%20Type%20III%20SS%20will,weighted%20squares%20of%20means%20analysis.)
- [Visualizing regularization and the L1 and L2 norms | by Chiara Campagnola | Towards Data Science](https://towardsdatascience.com/visualizing-regularization-and-the-l1-and-l2-norms-d962aa769932)

## Day 39: 8 Oct 2022

- TAMU datathon day
- `git gc` when facing hangups in git push

Links 🔗

- [TAMU Datathon](https://tamudatathon.com/)
- [benthecoder/gas_turbine_viz](https://github.com/benthecoder/gas_turbine_viz)
- [benthecoder/bloomberg_nlp](https://github.com/benthecoder/bloomberg_nlp)

## Day 40: 9 Oct 2022

- `R CMD build` to build R packages
- `R CMD install` to install R packages
- `tempdir()` to create temporary directory `tempfile()` to create temporary file

Links 🔗

- [Building and installing an R package](https://kbroman.org/pkg_primer/pages/build.html)
- [Temporary Storage in R | Jonathan Trattner](https://www.jdtrat.com/blog/temporary-storage-in-r/)

## Day 41: 10 Oct 2022

- learned about Fugue
- learned about adding progress bars to R with library(progress) and library(progressR) for furrr

Links 🔗

- [Fugue in 10 minutes — Fugue Tutorials](https://fugue-tutorials.readthedocs.io/tutorials/quick_look/ten_minutes.html)
- [r-lib/progress: Progress bar in your R terminal](https://github.com/r-lib/progress)
- [Progress notifications with progressr • furrr](https://furrr.futureverse.org/articles/progress.html)

## Day 42: 11 Oct 2022

- vectorization speed up in pandas
  - `np.where` for logical conditions
  - `np.select` for 2+ logical conditions
  - ndarray type casting is faster than .dt accessor for date values

Links 🔗

-[1000x faster data manipulation: vectorizing with Pandas and Numpy](https://www.youtube.com/watch?v=nxWginnBklU)

## Day 43: 12 Oct 2022

- worked on experts.ai nlp, yfinance and news api to build company monitoring app
- did statistics homework on blocking and factorial analysis

Links 🔗

- [How AI Image Generators Work (Stable Diffusion / Dall-E)](https://www.youtube.com/watch?v=1CIpzeNxIhU&t=28s)
- [Expert.ai](https://www.expert.ai/)

## Day 44: 13 Oct 2022

- factor analysis
  - what? taking a bunch of variables and reducing them to a smaller number of factors
  - goal : find latent variables
  - based on common factor model
  - rotation: diffferent ways to rotate the factors to make them more interpretable
  - ex: grades for many students
    - latent variables : language ability and technical ability
    - specific factors : measure impact of one specific measured variable on the latent variable (english skill on language and math skill on technical, etc.)
  - mathematical model
    - apply matrix decomposition to correlation matrix where diagonal entries are replaced by 1 - var(diagonal)
- distance measures in ML
  - Euclidean : shortest path between objects (l1 norm)
  - manhattan : rectilinear distance between objects (l2 norm)
  - minkowski : generalization of euclidian and manhattan, you can control which to use depending on data with h (L-h norm)
  - hamming : distance between two binary vectors / similarity measure for nominal data
  - cosine similarity : cosine of angle between two vectors, determines if two vectors point to similar directions

Links 🔗

- [What is the difference between PCA and Factor Analysis? | by Joos Korstanje | Towards Data Science](https://towardsdatascience.com/what-is-the-difference-between-pca-and-factor-analysis-5362ef6fa6f9)
- [4 Distance Measures for Machine Learning](https://machinelearningmastery.com/distance-measures-for-machine-learning/)

## Day 45: 14 Oct 2022

- active learning
  - idea: build better ml models using fewer labeled data by strategically choosing samples
  - process: pool-based active learning (uses "uncertainty sampling" to choose samples)
    1. label small percentage of data
    2. select samples randomly or strategically that provides most information
    3. train model, use it to predict labels for remaining samples
    4. use prediction and confidence in two ways
       1. add any samples with high confidence to labeled data (assumes high trust in data)
       2. valid approach is to skip first few iterations before automatically adding samples to labeled data
       3. request labels for samples with low confidence (crucial part of active learning)
       4. amount of information from high-confidence samples is low, low-confidence samples indicate our model needs help, so we focus labelling efforts on them.
    5. use improved labeled dataset to train new model and repeat process.
  - other approaches : stream-based selective sampling, membership query synthesis, etc.

Links 🔗

- [Active Learning | Bnomial](https://articles.bnomial.com/active-learning)

## Day 46: 15 Oct 2022

- hackwashu hackathon all day
- fast.ai nlp lesson 1

Links 🔗

- [What is NLP? (NLP video 1)](https://www.youtube.com/watch?v=cce8ntxP_XI)

## Day 47: 16 Oct 2022

- fast.ai nlp lesson 2
  - stop words: long considered standard techniques, but they can often hurt your performance if using deep learning. Stemming, lemmatization, and removing stop words all involve throwing away information. However, they can still be useful when working with simpler models.
  - SVD: singular value decomposition, used to reduce dimensionality of matrix
    - an exact decomposition, since the matrices it creates are big enough to fully cover the original matrix.
    - topics are orthogonal since words that appear frequently in one topic will appear rarely in another.
    - SVD is extremely widely used in linear algebra, and specifically in data science, including:
      - semantic analysis
      - collaborative filtering/recommendations (winning entry for Netflix Prize)
      - calculate Moore-Penrose pseudoinverse
      - data compression
      - principal component analysis
- started SVM chapter hands on ml
  - think of SVM as a linear model that tries to separate the classes by as wide a street as possible
  - SVMs are sensitive to feature scales, so you should always scale them
  - soft and hard margin classification
    - soft : allow some misclassifications, controlled by hyperparameter C (width of margin, higher C = narrower margin)
    - hard : no misclassifications

Links 🔗

- [Topic Modeling with SVD & NMF (NLP video 2)](https://www.youtube.com/watch?v=tG3pUwmGjsc&t=2142s)

## Day 48: 17 Oct 2022

- NMF (non-negative matrix factorization) fast.ai nlp lesson 2
  - SVD constraints factors to be orthogonal, NMF constraints them to be non-negative
  - factorization of non-negative dataset V -> non negative matrices W \* H
    - positive factors are more easily interpretable
    - negative factors in SVD does not make sense (ex: negative weight for word in topic)
- randomized svd faster than regular svd
  - randomized algorithms has advantages
    - inherently stable
    - matrix vector products can be computed in parallel

Links 🔗

- [NMF — A visual explainer and Python Implementation](https://towardsdatascience.com/nmf-a-visual-explainer-and-python-implementation-7ecdd73491f8)
- [Randomized Singular Value Decomposition](https://gregorygundersen.com/blog/2019/01/17/randomized-svd/)

## Day 49: 18 Oct 2022

- NLP week 3
  - factorization and matrix decomposition
    - multiplication 2 x 2 = 4
      - prime factorization = decomposing intergers into something else
        - nice property of factors being prime
        - harder than multiplication (heart of encryption)
      - think of factorization as the opposite of multiplication
    - matric decomposition
      - a way of taking matrices apart and come up with matrices with nice properties
      - what are nice propertices of matrices in SVD?
        - A = USV
        - U (columns) and V (rows) are orthonormal (orthogonal to each other and pairwise normalized)
        - S is diagonal matrix with singular values (captures how important each factor is in descending order)
      - nice properties of NMF
        - non negative
        - often end up being sparse

Links 🔗

- [Topic Modeling & SVD revisited (NLP video 3)](https://www.youtube.com/watch?v=lRZ4aMaXPBI)

## Day 50: 19 Oct 2022

- [Iterative](https://iterative.ai/)
  - [DVC](https://github.com/iterative/dvc): Data Version Control (ML Experiments, Pipelines, Git for data)
    - data versioning : large datasets are replaced with metafiles that point to the actual data
    - data storage: on-premises or cloud storage can be used
    - reproducible: pipelines, dependency graphs and codified data and artifacts
  - [CML](https://github.com/iterative/cml): Continuous Machine Learning is CI/CD for Machine Learning Projects
    - automate model training and evaluation
    - ml experiment
    - changing datasets
- pytest
  - fixtures and mocks

Links 🔗

- [Iterative](https://github.com/iterative)
- [Data Version Control · DVC](https://dvc.org/doc/user-guide/overview)
- [CML · Continuous Machine Learning](https://cml.dev/doc/usage)
- [How Data Scientists Can Improve Their Productivity | Iterative](https://iterative.ai/blog/how-data-scientists-can-improve-their-productivity)
- [htorrence/pytest_examples: Reference package for unit tests](https://github.com/htorrence/pytest_examples)
- [Unit Testing for Data Scientists - Hanna Torrence](https://www.youtube.com/watch?v=Da-FL_1i6ps)

## Day 51: 20 Oct 2022

- Clustering
  - distance measures
    - common: euclidian and manhattan
    - euclidian: (not robust to outlires)
    - Manhattan, Minkowski, or Gower distances (robust to outlieres)
      - Gower distance measure: mixed features
    - correlation-based (good for data with large magnitude differences but similar behaviour like [customer purchases](https://bradleyboehmke.github.io/HOML/kmeans.html#fig:correlation-distance-example))
    - cosine distance: text data
  - k means
    - algo: pick k random centroids, assign each instance to the closest centroid, move the centroids to the center of the instances that were assigned to them, repeat until convergence. cost function is the sum of squared distances between each instance and its closest centroid.
    - what's the right K?
      - predetermined by domain expert
      - k = sqrt(n/2) (not good for large datasets)
      - elbow method: plot cost function vs k, pick k where cost function starts to flatten out
  - hierarchical clustering
    - creates hierarchy of clusters
    - advantage: does not require you to specify the number of clusters and creates a dendrogram, however at the end you still have to decide where to cut the dendrogram to get the number of clusters you want
    - two types:
      - agglomerative (AGNES): bottom up approach, start with each instance in its own cluster, then merge the two closest clusters, repeat until all instances are in one cluster
      - divisive (DIANA): top down approach, start with all instances in one cluster, then split the cluster into two, repeat until each instance is in its own cluster
    - determine number of clusters
      - gap statistics: maximum gap between the log of the average within-cluster sum of squares and the log of the average between-cluster sum of squares
  - model-based clustering
    - previous methods directly derives from data, model-based incoroprates a measure of probability or uncertainty to the cluster assignments
    - provides soft assignment of instances to clusters (probability of each instance belonging to each cluster) and automatically determines the number of clusters
    - key idea: data are considered as coming from a mixture of underlying probability distributions
    - The most popular approach is the Gaussian mixture model (GMM), wehre each obs is assumed to be distributed as one of k-multivariate normal distributions
    - idea of probabilistic cluster assignment is useful as you can identify observations with high or low cluster uncertainty. Ex: marketing, if person is 50% being assigned to two clusters, provide them with combination of marketing solutions for both clusters.
    - how to capture different structures of cluster?
      - covariance matrix describes the geometry of the clusters; namely, the volume, shape, and orientation of the clusters, which allows GMMs to [capture different cluster structures](https://bradleyboehmke.github.io/HOML/model-clustering.html#fig:visualize-different-covariance-models)
    - limitations
      - require underlying distribution to be normal, results are heavily dependent on that assumption
      - computationally expensive in high dimensional spaces, problem is it's heavily overparameterized. solution is to do dimensionality reduction, but that can lead to loss of information

Links 🔗

- [Clustering | Hands-On Machine Learning with R](https://bradleyboehmke.github.io/HOML/kmeans.html)

## Day 52: 21 Oct 2022

- Working on Notion NLP

Links 🔗

- [benthecoder/notion_nlp: Adds NLP to Notion](https://github.com/benthecoder/notion_nlp)

## Day 53: 22 Oct 2022

- makeUC hackathon

Links 🔗

- [weichunnn/connectify](https://github.com/weichunnn/connectify)

## Day 54: 23 Oct 2022

- TSNE
  - goal: is to take a set of points in high-dimensional space and find a faithful representation of those points in a lower-dimensional space, typically 2d
  - it's incredibly flexibly and you can find structure where other dim. reduction techniques fails
  - "Perplexity"
    - a tunable parameter which is a guess about the number of neighbors each point has
    - this parameter has a complex effect and may be more nuanced than just setting typical values between 5 and 50
    - Getting the most from t-SNE may mean analyzing multiple plots with different perplexities.
  - most important thing: there is no fixed steps to yield a stable result, different datasets require different number of iterations
  - note:
    - cluster sizes in a t-SNE plot are not indicative of the actual cluster sizes in the original data
    - distances between well-separated clusters in a t-SNE plot may mean nothing
    - at low perplexities, you might see dramatic clusters, but they are random noise and meaningless

Links 🔗

- [How to Use t-SNE Effectively](https://distill.pub/2016/misread-tsne/)

## Day 55: 24 Oct 2022

- JSON is slow
  - `"type" : 0"` takes 9 bytes to represent single byte (4 for key, 2 for quotes, 1 for colon, 1 for comma)
  - `"timestamp" : 1666295748665` takes 27 bytes when it could just be `95748665`
  - alternative? [deku](https://docs.rs/deku/latest/deku/)
  - "JSON is like the python of language, slow but convenient to use"
- Python 3.11 new features
  - 10-60% faster than Python 3.10
  - better error messages
- Hypothesis testing

Links 🔗

- [Is JSON Blazingly Fast or...?](https://www.youtube.com/watch?v=MuCK81q1edU)
- [It's time... for PYTHON 3.11!](https://www.youtube.com/watch?v=ueTXYhtlnjA)
  - [code](https://github.com/mCodingLLC/VideosSampleCode/tree/master/videos/109_python_311_release)
- [Probably Overthinking It: There is still only one test](https://allendowney.blogspot.com/2016/06/there-is-still-only-one-test.html)
- [John Rauser keynote: "Statistics Without the Agonizing Pain" -- Strata + Hadoop 2014](https://www.youtube.com/watch?v=5Dnw46eC-0o)

## Day 56: 25 Oct 2022

- precision: TP / (TP + FP) (how many of the positive predictions were correct)
  - high precision = cost of false negatives, failing to identify who has the disease
- recall: TP / (TP + FN) (how many of the positive cases were predicted correctly)
  - high recall = cost of false positives, misdiagnose people who don't have the disease
- F1-score : harmonic mean of precision and recall
  - completely ignores true negatives, misleading for unbalanced classes
- Many other techniques exis for model evaluation, including calibration, popular diagnostic tools (specificity, likelihood ratios, etc.), and expectation frameworks
- ROC curve = True-Positive Rate (TPR) versus False-Positive Rate (FPR)
  - It allows us to understand how that performance changes as a function of the model's classification threshold
- AUC ranges in value from 0 to 1, with higher numbers indicating better performance. A perfect classifier will have an AUC of 1, while a perfectly random classifier an AUC of 0.5
- probabilistic interpretation
  - The AUC is the probability that the model will rank a randomly chosen positive example more highly than a randomly chosen negative example.
- the AUC metric treats all classification errors equally, ignoring the potential consequences of ignoring one type over another

Links 🔗

- [Precision & Recall](https://mlu-explain.github.io/precision-recall/)
- [ROC & AUC](https://mlu-explain.github.io/roc-auc/)

## Day 57: 26 Oct 2022

- correlation matrix = standardized covariance matrix
- PCA
  - a method of finding low-dimensional representation of data that retains as much original variation as possible
  - in PCA, each pc are a linear combination of the original p features
  - reducing features to latent (unobserved) variables can remove multicollinearity and improve predictive accuracy
  - first PC captures the largest amount of variance in the data, it is simply a line that minimizes teh total squared distance from each point to its orthogonal projection onto the line, second PC is orthogonal to first, a linear combination of features that has maximal variance that is uncorrelated with first PC
  - how are PCs calculated?
    - eigenvector corresponding to largest eigenvalue of feature covaraince matrix is teh set of loadings that explains the greatest proportion of feature variability
    - recap: eigenvector = vector that doesn't change direction when a linear transformation is applied to it, eigenvalue = scalar that tells you how much the eigenvector is stretched or squished by the linear transformation
  - standardize (using correlation) when
    - different variables have greatly different variances
    - you want more emphasis on describing correlation than variance
  - scores are measure of contribution of the variable to the variation of PC
  - to interpret scores, look at both size and sign
- Factor Analysis
  - "inversion" of PCA
  - we model observed variables as linear functions of the "factors"
  - in pca we create new variables that are linear combinations of the observed variables
  - in PCA, interpretation is not very clean as one variable can contribute significantly to more than one of the components
  - with factor rotation, each variable contributes significantly to only one component
- PCA vs Factor Analysis
  - similar: both are linear methods for reducing dimensionality, both transform data linearly, both claim we can use few linear combinations to explain most of the variation in the data
  - difference:
    - PCs are ordered from highest to lowest information, retains only a few PCs
    - aims to find latent variable that make sense, loadings are rotated to improve interpretability. it also estimates specific effects (factors)
  - use PCA to reduce dimensionality, prepare data
  - use Factor Analysis to find latent variables, interpret data

Links 🔗

- [6. Singular Value Decomposition (SVD) - YouTube](https://www.youtube.com/watch?v=rYz83XPxiZo)
- [Principal Component Analysis explained visually](https://setosa.io/ev/principal-component-analysis/)
- [Eigenvectors and Eigenvalues explained visually](https://setosa.io/ev/eigenvectors-and-eigenvalues/)
- [Lesson 12: Factor Analysis | STAT 505](https://online.stat.psu.edu/stat505/lesson/12)

## Day 58: 27 Oct 2022

1. Draw more pictures
   - draw diagrams of architecture, define notations for different domains, create modeling tools
2. Review at the right level
   - make design review part of the process, run the code, promote use of patterns, dont obsess over syntax, focus on algorithms
3. Value stability
   - write quality documentation, take deep interest into how frameworks work
4. Invent it here
   - take control of dependencies and evalutate cost/benefit of external libraries
5. Learn to test
   - understand whole scope of testing, not all tests can be automated, take non-functional test seriously
6. Master the tools
   - learn keyboard shortcuts, master IDE, be comfortable with terminal, automate boring stuff
7. Focus on fundamentals
   - focus on re-usable skills, algorithms, DS, clean code, design principles, avoid spending time (re) learnign next shiny framework
8. Remain accountable
   - focus principles over process - understand the why, be professional, take responsiblity and be accountable for your work
9. Prepare for rain
   - think about career, educate yourself, stay in demand, prepare for extreme changes (downturns, layoffs, etc.)
10. Remember what matters
    - time passes more quickly than you think, enjoy the journey, be kind to yourself, be kind to others

Links 🔗

- [10 Programming Lessons From the Past • Garth Gilmour & Eamonn Boyle](https://www.youtube.com/watch?v=IlDIV5gaTP0&t=1463s)
- [The Art of Visualising Software Architecture - YouTube](https://www.youtube.com/watch?v=zcmU-OE452k)
- [Testing quadrant](https://medium.com/yoursproductly/61-why-pms-must-understand-the-agile-testing-quadrants-to-help-plan-their-product-qa-efforts-710ba6356002)

## Day 59: 28 Oct 2022

- how to learn
  - learn about yourself (what do you like, what allows you to enter a state of flow?)
  - acquire those skills (choose to do the same thing over and over again for 10,000 hours until you become great at it)

Links 🔗

- [LEARN, EARN or QUIT | My job/career advice for 2022 | Garry Tan's Founders Journey Ep. 4 - YouTube](https://www.youtube.com/watch?v=eLelgy5zRv4&t=557s)

## Day 60: 29 Oct 2022

- Feature selection
  - filter methods (ranking methods)
    - process: rank features -> select highest ranking features
    - pros: model agnostic and fast
    - cons: ignores feature redundancy, feature interaction and feature-model interaction
    - examples
      - chi-square : categorical vars and cat. target
      - anova : continuous vars and cat. target
      - correlation : continuous vars and continuous target
      - mutual information
      - variance
  - wrapper methods
    - process: create subset feature -> train model on subset -> get model performance -> select best subset
    - pros: feature model and feature interaction
    - cons : not model agnostic and computationally expensive
    - examples:
      - exhaustive search, forward search, backward search
  - embedded methods
    - process: train model -> derive feature importance -> remove non-important features
    - pros : fast and captures feature interaction
    - cons: limited to some models, not model agnostic
    - examples
      - lasso
      - tree derived feature importance
      - regression coefficients
  - other methods
    - feature permutation, probe features, mRMR (minimum-Redundancy-Maximum-Relevance), Recursive Feature Elimination (RFE), cluster based feature selection (CBFS),
    - statistics: population stability index, information value

Links 🔗

- [Grokking Stable Diffusion.ipynb - Colaboratory](https://colab.research.google.com/drive/1dlgggNa5Mz8sEAGU0wFCHhGLFooW_pf1?usp=sharing)
- [solegalli/DataTalks.Club2022](https://github.com/solegalli/DataTalks.Club2022)
- [Feature-engine — 1.3.0](https://feature-engine.readthedocs.io/en/1.3.x/index.html)

## Day 61: 30 Oct 2022

- ANOVA
  - robustness
    - robust against departures from normality when sample size is large enough
    - robust against unequal variances when sample sizes are equal
  - type I and II
    - type I (alpha) = probability of rejecting the null hypothesis when it is true
    - type II (beta) = probability of failing to reject null hypothesis when it is false
    - power (1 - beta) = probability of correctly rejecting null hypothesis when it is false
    - best case : low alpha and high power
  - anova assumptions
    - normality
      - if violated: power (ability to detect true difference) decreases
      - evaluation: QQ plot and histogram of residuals
    - homogeneity of variance
      - if violated and sample size are unequal: type I increases if smaller group has larger varaince and power decreases if larger group has larger variance
      - evaluation: residual plot / statistical tests - Bartlett (inaccurate if slightly nonnormal), Brown-Forsythe (best but can be resource intensive with large groups), Levene (standard test), and O’Brien (modified Levene)
    - independence
      - if violated: type I error rate increase (if positive correlation) and power decrease (if negative correlation)
      - evaluation: know source of data, examples of correlated data : complex survey designs, repeated measures, data gathered over time
- PETS = privacy-enhancing technologies
  - based on Yao's "millionaires problem" - is it possible for two millionares to discover who is richer without revealing their wealth?
  - "secure multi-party computation" - Allows two or more parties to compute on their shared data, without any party revealing any of their private data.
  - "zero-knowledge proofs" - Allows a person to prove to another person that they know something is true, without revealing any information on how they know it is true.
  - "fully homomorphic encryption" - holy grail of cryptography, in which it is possible to run analytics on encrypted data without decrypting it first
    - ex: register password: password is encrypted -> sent to server to check whether it has been breached without server being able to identify the password
  - "differential privacy" - add noise to data to preserve privacy of individuals
    - ex: official statistics: simple averages can reveal private info about people from minority groups

Links 🔗

- [Python Virtual Environment & Packaging Workflow | Peter Baumgartner](https://www.peterbaumgartner.com/blog/python-virtual-environment-package-workflow/)
- [50 Useful Vim Commands - VimTricks](https://vimtricks.com/p/50-useful-vim-commands/)
- [Can a new form of cryptography solve the internet’s privacy problem? | Data protection | The Guardian](https://www.theguardian.com/technology/2022/oct/29/privacy-problem-tech-enhancing-data-political-legal)

## Day 62: 31 Oct 2022

- split plot design
  - special case of factorial treatment structure
  - used when some factors are harder (more expensive) to change than others
  - consists of two experiments with different experimental units of different "sizes"

Links 🔗

- [14.3 - The Split-Plot Designs | STAT 503](https://online.stat.psu.edu/stat503/lesson/14/14.3)

## Day 63: 1 Nov 2022

- factorial design
  - experiemnt with two more more factors, each with discrete possible levels and whose experimental units take on all possible combinations of levels of the factors
  - experimental units (eu) is the entity a researcher wants to make inferences about, e.g. a person, a group, a company, a country
  - vast majority of factorial experiments, each factor has two levels, with two factors taking two levels, there would be 4 treatment combinations in total - 2x2 factorial design
  - however experimenting with 10 factors at two levels produces 2^10 = 1024 treatment combinations, which becomes infeasible due to high cost, in these cases, use fractional factorial designs
  - full factorial design
    - what?: all possible combinations of factors
    - how? : each factor has two levels (A and B) and each level is assigned to a group of experimental units
    - why? : to determine the effect of each factor on the response variable
  - fractional factorial design
    - what?: subset of all possible combinations of factors
    - how? : random sampling, orthogonal arrays, response surface methods
    - why? : reduce number of experiments, reduce cost, reduce time

Links 🔗

- [5.3.3. How do you select an experimental design?](https://www.itl.nist.gov/div898/handbook/pri/section3/pri33.htm)
- [Design of Experiments](https://online.stat.psu.edu/stat503/)
- [CRAN Task View: Design of Experiments (DoE) & Analysis of Experimental Data](https://cran.r-project.org/web/views/ExperimentalDesign.html)
- [Experimental Design and Analysis](https://www.stat.cmu.edu/~hseltman/309/Book/Book.pdf)
- [Design and Analysis of Experiments with R](http://www.ru.ac.bd/stat/wp-content/uploads/sites/25/2019/03/502_07_00_Lawson_Design-and-Analysis-of-Experiments-with-R-2017.pdf)
- [Design of Experiments | Coursera](https://www.coursera.org/specializations/design-experiments)

## Day 64: 2 Nov 2022

- Expected means square (EMS)
  - what? : expected values of certain statistics arising in partitions of SS in ANOVA
  - used for determining which statistics should be in denominator in F-test for testing a null hypothesis that a particular effect is absent
  - note:
    - test statistic = ratio of two mean squares where the E(numerator) differes from E(denominator) by the variance component or fixed factor we're interested in. Thus, under assumption of H0, both numerator and denominator of F ratio has same EMS
    - varaince component = the variance of the response variable due to a particular factor
- basic principles of DOE
  - randomization
    - essential component of any experiemnt to have validity. assigning treatments to experimental units has to be a random process when doing a comparative experiment to eliminate potential biases
  - replication
    - heart of all statistics, more replicates = more precise estimates. ex: confidence interval width depends on standard error = sqrt(s^2/n), to reduce it we need to reduce error in estimate of mean by increasing n (estimate becomes less variable as sample size increases CLT). The other (harder) option is reduce error varaince, this is where blocking comes in
  - blocking
    - main idea is to make unknown error variance to be as small as possible, goal is to find out about treatment factor, but in addition to this, we want to include any blocking factors that can explain away the variation.
  - multi-factor design
    - one factor at a time is inefficient, multi-factor : 2k, 3k, rsm, etc. includes combination of multiple factors. You learn not only primary factors but also other factors - interactions and confounding factors
  - confounding
    - typically avoid, but in complex experiments, confounding can be used to advantage. ex: interested in main effects but not interactions -> confound interactions to reduce sample size -> reduce cost of experiment but still have good info on main effects

Links 🔗

- [13.4 - Finding Expected Mean Squares | STAT 503](https://online.stat.psu.edu/stat503/lesson/13/13.4)
- [Applied Statistics: STAT 500 notes](https://online.stat.psu.edu/stat500/)
- [1.2 - The Basic Principles of DOE | STAT 503](https://online.stat.psu.edu/stat503/lesson/1/1.2)

## Day 65: 3 Nov 2022

- stat 475
  - multidimensional scaling
    - an attempt to make a 2d representation of how "close" different "things" are.
    - a means of visualizing the level of similarity of individual cases of dataset
    - used to translate information about the pairwise distances among a set of objects or individuals into a configuration of points mapped into an abstract Cartesian space
    - technical: a set of related ordination techniques used in information visualization, in particular to display the information contained in a distance matrix.
    - given a distance matrix with the distances between each pair of objects in a set, and a chose number of dimensions, N, an MDS algo places each object into an N-dim space such that the between-object distances are preserved as well as possible. For N = 1, 2, 3 it can be visualized on a scatterplot.
  - hotelling T^2
    - what? : a test statistic used to test whether the mean of a multivariate normal distribution is equal to a given vector
    - how? : T^2 = (x - mu)' \* S^-1 \* (x - mu) where x is a sample mean vector, mu is a hypothesized mean vector, S is a covariance matrix
- notes on writing well
  - Occam's razor: the shorter writing is usually better, as it is clearer and more vivid. e.g. "big" is clearer than "humungous"
  - Optimize for the truth: early writings are always rough approximations to the truth, revise over and over and ask whether each sentence can be sharper and if we truly believe the sentnece
  - good writing = good psychology: good writing is a relationship between the reader and the text, useful advice is about how to change that relationship. good writers build up a theory of that relationship. Good writing is really an expercise in applied psychology

Links 🔗

- [notes-on-writing/notes_on_writing.md](https://github.com/mnielsen/notes-on-writing/blob/master/notes_on_writing.md)
- [Multidimensional Scaling](https://towardsdatascience.com/multidimensional-scaling-d84c2a998f72)
- [7.1.3 - Hotelling’s T-Square | STAT 505](https://online.stat.psu.edu/stat505/lesson/7/7.1/7.1.3)

## Day 66: 4 Nov 2022

- NA Connect 2022 Day 1
  - tensorflow lite
  - similarity based machine learning

Links 🔗

- [North America Connect 2022](https://rsvp.withgoogle.com/events/na-connect-2022)
- [Machine Learning Explainability Workshop I Stanford - YouTube](https://www.youtube.com/playlist?list=PLoROMvodv4rPh6wa6PGcHH6vMG9sEIPxL)

## Day 67: 5 Nov 2022

- NA Connect 2022 Day 2

Links 🔗

## Day 68: 6 Nov 2022

- problem with CSV files
  - they are slow to query: SQL and CSV do not play well together.
  - they are difficult to store efficiently: CSV files take a lot of disk space.
- CSV are row-oriented while parquet is column oriented
- why is it better than row storing?
  - parquet files are much smaller (compressed column by column)
    - A CSV file of 1TB becomes a Parquet file of around 100GB (10% of the original size.
  - parquet files are much faster to query
    - sql query selects and aggregates subset of columns do you don't have to scan other columns, reduces I/O
  - parquet files are cheaper to store in cloud storage

Links 🔗

- [Forecasting something that never happened: how we estimated past promotions profitability](https://medium.com/artefact-engineering-and-data-science/forecasting-something-that-never-happened-how-we-estimated-past-promotions-profitability-5f55cfa1d477)
- [A Parquet file is all you need | Towards Data Science](https://towardsdatascience.com/a-parquet-file-is-all-you-need-962df86886bb)

## Day 69: 7 Nov 2022

- set_output(transform = "pandas") to get pandas output from sklearn
- What is the distributional hypothesis in NLP? Where is it used, and how far does it hold true?
  - According to the distributional hypothesis, words occurring in the same contexts tend to have similar meanings (original source: Distributional Structure by Zellig S. Harris, 1954). In other words, the more similar the meanings of two words are, the more often they appear in similar contexts. So, for example, the words cat and dog often occur in similar contexts and are more related (both are mammals and pets) than a cat and a sandwich. Looking at large datasets, this may hold more or less, but it is easy to construct individual counter-examples.
  - The distributional hypothesis is the main idea behind Word2vec, and many natural language transformer models rely on this idea (for example, the masked language model in BERT and the next-word pretraining task used in GPT).
- What is the difference between stateless and stateful training? And when to use which
  - Both stateless (re)training and stateful training refer to different ways of training a production model. Stateless training is like a sliding window that retrains the model on different parts of the data from a given data stream.
  - In stateful training, we train the model on an initial batch of data and then update it periodically (as opposed to retraining it) when new data arrives.
  - One paradigm is not universally better than the other. However, an advantage of stateful training is that it doesn’t require permanent data storage. On the other hand, storing data, if possible, is still a good idea because the model might suffer from “bad” updates, and (temporarily) switching to stateless retraining might make sense.
- What is the difference between recursion and dynamic programming?
  - In recursion, we divide a problem into smaller subproblems in an iterative fashion, often called “divide-and-conquer.”
  - Dynamic programming avoids computing solutions to the same subproblems by storing solutions to subproblems in a data structure that allows fast (constant time) look-ups, for example, dictionaries. Storing the subproblems is also often called “memoization” (not to be confused with “memorization”).
  - In practice, we often apply dynamic programming to recursive algorithms.

Links 🔗

- [Ahead of AI #2: Transformers, Fast and Slow | Revue](https://newsletter.sebastianraschka.com/issues/ahead-of-ai-2-transformers-fast-and-slow-1402662)
- [koaning/embetter: just a bunch of useful embeddings](https://github.com/koaning/embetter?utm_campaign=Ahead%20of%20AI&utm_medium=email&utm_source=Revue%20newsletter)
- [[2210.06280] Language Models are Realistic Tabular Data Generators](https://arxiv.org/abs/2210.06280?utm_campaign=Ahead%20of%20AI&utm_medium=email&utm_source=Revue%20newsletter)
- [A Short Chronology Of Deep Learning For Tabular Data](https://sebastianraschka.com/blog/2022/deep-learning-for-tabular-data.html?utm_campaign=Ahead%20of%20AI&utm_medium=email&utm_source=Revue%20newsletter)

## Day 70: 8 Nov 2022

- multiple linear regression recap
  - coefficeints: change in Y associated with 1-unit chang ein x1 while holding other predictors constant
  - categorical predictors that are nominal should be indicator variables, cannot impose order
  - R^2: how well regression mode explains observed data
  - SSE: how much variation in observed y values is not explained by model relationship
  - adjusted r^2: penalizes for adding predictors that don't improve model, if adjusted r^2 is a lot lower than r^2, then the model is overfitting
  - model utility: are all predictors useful? a join test based on F distribution, testing if all coefficients are 0
  - anova on reduced and full model: if p-value is low, then the model is useful, if p-value is high, then the model is not useful
  - multicollinearity: predictors are correlated with each other, can be detected by looking at correlation matrix
    - what? it's not an error, but a lack of information in dataset
    - how to detect? plots and correlation tables, significant F-statistic for overall test of model but no single predictor is significant, estimated effect of covariate has opposite sign of what you expect

Links 🔗

- [Lesson12_MultRegression](https://www.colorado.edu/amath/sites/default/files/attached-files/lesson12_multregression.pdf)
- [10.8 - Reducing Data-based Multicollinearity | STAT 462](https://online.stat.psu.edu/stat462/node/181/)

## Day 71: 9 Nov 2022

- transformers
  - use cases
    - language translation : Meta's [No Language Left Behind](https://ai.facebook.com/blog/nllb-200-high-quality-machine-translation/?utm_campaign=Ahead%20of%20AI&utm_medium=email&utm_source=Revue%20newsletter), a transformer model capable of translating between 200 languages.
    - protein folding : alphaFold2 model is based on the language transformers, Meta released [ESM Metagenomic Atlas](https://esmatlas.com/explore?at=1%2C1%2C21.999999344348925), 60x times fster than AlphaFold2
    - question answering : better chatbots, next gen Q&A systems
  - AGI?
    - mechanisms that help transformers to self-improve via a pre-training algorithm distillation based on RL
  - environmental cost
    - BLOOM, a 176B parameter transformer model, required 1,082,990 GPU hours (on cutting edge Nvidia A100s) and emitted approximately 50.5 tonnes of CO2 in total
  - resources

Links 🔗

- [Attention Is All You Need](https://www.explainpaper.com/papers/attention)s
- [[2210.11610] Large Language Models Can Self-Improve](https://arxiv.org/abs/2210.11610)
- [amazon-science/mintaka: Dataset from the paper "Mintaka: A Complex, Natural, and Multilingual Dataset for End-to-End Question Answering" (COLING 2022)](https://github.com/amazon-science/mintaka)
- [[2211.02001] Estimating the Carbon Footprint of BLOOM, a 176B Parameter Language Model](https://arxiv.org/abs/2211.02001)
- [Introduction to Deep Learning](https://sebastianraschka.com/blog/2021/dl-course.html)
- [(1) 2022.02 Transformers - Lucas Beyers](https://www.youtube.com/watch?v=UpfcyzoZ644)
- [karpathy/minGPT: A minimal PyTorch re-implementation of the OpenAI GPT (Generative Pretrained Transformer) training](https://github.com/karpathy/minGPT)

## Day 72: Nov 10, 2022

- NLP tricks by Vincent Warmerdam
  - `CountVectorizer(analyzer="char", ngram_range= (2, 3))` helps capture more information and context and makes it more robust against spelling errors as it looks at characters
  - partial_fit on sklearn helps working with larger datasets
  - BLOOM embeddings by explosion.ai
    - hashing trick to reduce size of embedding
  - bulk labelling
    - text/image -> n dimensional -> 2d (UMAP) -> text and image interface with bulk
- block websites on mac
  - `sudo /etc/hosts`
  - add 127.0.0.1 www.website.com
  - remove cache `sudo dscacheutil -flushcache`
- Pinterest ML day 2022
  - PinSAGE: Graph NN with 3 billion nodes and 18 billion edges produces amazing content represetation
    - pin -> multi-modal embedding -> system
  - ml applications
    - Personalization / Recommendations - ranking, retrieval, etc.
    - User Interest Modeling
    - Ads - Targeting, Pricing, Relevance, etc.
    - Content Understanding - Text, Computer Vision, Multimodal
    - Notifications - Content and Scheduling
    - Visual Search - Pinterest Lens
    - Inclusive Al - Skintone and Hair Pattern result diversification and filtering
    - Creating New Content - Virtual Try On, Image Composition
  - ML modeling technology
    - Transformers - Sequence Modeling, Content Understanding, User Understanding, Computer vision - embeddings
    - Graph Neural Networks - Content Understanding
    - Real Time Sequence Modeling
    - Generative Models - Diffusion Models
    - Two Tower Models - for retrieval
    - Multi-Task ML
    - Large Scale Models - 1B++ parameters
  - ml infra
    - \>1 exabyte of data and >400M inferences / second
    - MLEnv - GPU Inference Acceleration: 100x speed-up using GPU serving comparing to single CPU
    - Training
      - TCP - Training Compute Platform
      - Training Data Engine
      - Centralized Feature Store, Unified Feature Representation
      - Automatic Retraining Framework
  - to come
    - Representation Learning for text, videos, products, creators, search queries, notifications.
    - Web Mining through GNNs to extract attributes (e.g. recipe for food pins) from websites to create rich content at scale.
    - Inspirational Knowledge Graph to enable a vocabulary to communicate between ML and users to assist their journey.
    - Learned Retrieval to holistically learn candidate generation for recommendations and search.
    - Notification Uplift Modeling to learn the optimal intervention policy for share inspiration to Pinners outside of Pinterest.

Links 🔗

- [Tricks and Tools From NLP-Land? - Vincent D. Warmerdam | Munich NLP Hands-on 008 - YouTube](https://www.youtube.com/watch?v=sjiASMMbHao)
- [koaning/embetter: just a bunch of useful embeddings](https://github.com/koaning/embetter)
- [partial_fit: Introduction](https://calmcode.io/partial_fit/introduction.html)
- [Compact word vectors with Bloom embeddings · Explosion](https://explosion.ai/blog/bloom-embeddings)
- [MaartenGr/BERTopic](https://github.com/MaartenGr/BERTopic)
- [koaning/bulk: A Simple Bulk Labelling Tool](https://github.com/koaning/bulk)
- [PinSage: A new graph convolutional neural network for web-scale recommender systems](https://medium.com/pinterest-engineering/pinsage-a-new-graph-convolutional-neural-network-for-web-scale-recommender-systems-88795a107f48)

## Day 73: Nov 11, 2022

- parallel processing in Python
  - ray
  - dask
  - spark

Links 🔗

- [Aaron Richter- Parallel Processing in Python| PyData Global 2020 - YouTube](https://www.youtube.com/watch?v=eJyjB3cNIB0)
- [Ray: A Framework for Scaling and Distributing Python & ML Applications - YouTube](https://www.youtube.com/watch?v=LmROEotKhJA)
- [Introduction to Distributed Computing with the Ray Framework - YouTube](https://www.youtube.com/watch?v=cEF3ok1mSo0)
- [Peter Baumgartner: Applied NLP: Lessons from the Field (spaCy IRL 2019) - YouTube](https://www.youtube.com/watch?v=QRGMJWwOU94)

## Day 74: Nov 12, 2022

- working on gpt3 blog title optimizer
- ads ranking models
  - recommender system basics:
    - set of features for user U and set of features for item I along features C capturing context of time of recommendation (time of day, weekend, etc.) and match those features to affinity event (did user click ad or not)
    - click/no click = F(U, I, C)
  - models
    - GBMs
      - good with dense features (age, gender,) but bad with sparse features (id of user, id of ad)
      - Those sparse features are categorical features with literally billions of categories and very few sample events. ex: consider the time series of sequence of pages visited by a user, how do you build features to capture that information?
      - solution: a page id becomes a vector in an embedding and a sequence of page Ids can be encoded by transformers as a simple vector. And even with little information on that page, the embedding can provide a good guess by using similar user interactions to other pages.
    - Multi-task learning
    - mixture of experts
    - multi-towers models
  - stages of ads ranking
    - first select a sub-universe of ads (1m ads)that relate to user (fast retrieval)
    - select subset of ads (1000 ads) with simple model (fast inference)
    - use complex model (slow inference) to rank resulting ads as accurately as possible
    - top ranked ad is the ad you see on your screen
    - use MIMO (multiple input, multiple output) model to simultaneously train simple and complex model for effecient 2 stage ranking

Links 🔗

- [1706.05098.pdf](https://arxiv.org/pdf/1706.05098.pdf)
- [Recommending What Video to Watch Next: A Multitask Ranking System](https://daiwk.github.io/assets/youtube-multitask.pdf)
- [Cross-Batch Negative Sampling for Training Two-Tower Recommenders](https://arxiv.org/pdf/2110.15154.pdf)
- [All Words - Machine Learning Glossary](https://machinelearning.wtf/all/)

## Day 75: Nov 13, 2022

- databricks vs snowflake
  - databricks
    - data storage interface + dev environment + scheduler + spin up compute with isolated dependencies
    - access mounted file systems
    - well executed paid version of spark with lots of stuff on top
  - snowflake
    - a database solution + query with SQL
- lessons
  - the most important attribute for a star data scientist is judgment.
  - The fundamental mission of any data science role is not to build complex models, but how to add value to the business
  - primary function of data science is to support business function — generate growth, improve engagement, reduce inefficiency, and ultimately, more revenue and less cost. This means:
    - You have to understand where the business is going (company’s North Star).
    - You have to understand how various business units such as marketing, product and engineering are currently working towards the company’s north star, and what their road maps are. Understand alignments and gaps, and how data can drive synergy.
    - You have to understand how the data science function fits into the business organically, and how the team is developing and contributing. Where are the biggest deficiencies both technically and operationally (data quality? pipeline efficiency? Communication clarity? Value?).
    - Then finally, how to get data, what you can learn from the data, what data product you can build cross functionally, and how that may change the business.
  - The biggest risk of many data scientists is working in the vacuum — trying too hard to solve techinical problems without thinking about the business impact.
  - The Why question is much more critial than How, and very often a 80% solution is superior to a 90% solution
  - And on top of the techinical decision making — you should be driving discussions, align teams, provide solid evidence supporting your perspectives about the past and the future
  - you also need very solid techinical foundation to deliver trustworthy findings. The crediblity of data scientist is everything — one major fallacy would lead to compromised trust, consequently banishment from the core decision tables
  - Very often the model performance metric is not the deciding factor of whether you should proceed for production. The potential incremental gain, cost of development time, required resources, implementation complexity, computation time, interpretability, data availability, fail-safe plans, impact to other teams… and many other factors are much more important
  - you need to be passionate about the business. You don’t come in and just explore data or build models for the sake of doing them, but you want to make the business better with your outstanding work.
  - you want to ask yourself where you want to be first. Different companies have very different needs, since you can’t do everything from the beginning, you need to focus — whether it’s time series predictions, natural language processing, recommendation systems or image classifications, you need to figure out what the business is looking for, and make alignments with your personal goals

Links 🔗

- [What Does Databricks Do?](https://interconnected.blog/what-does-databricks-do/)
- [A Random Mixbag Statistical Methods](https://medium.com/@m.sugang/a-random-mixbag-statistical-methods-74fbbfe8c3ac)
- [What I’ve Learned as a Data Scientist | by Gang Su | Medium](https://medium.com/@m.sugang/what-ive-learned-as-a-data-scientist-edb998ac11ec)

## Day 76: Nov 14, 2022

- Testing in Data Science
  - assert statements
    - on intermediate calculations or processes and basic calculation and arithmetic
    - examples
      - check symmetric difference between two sets are 0 when merging two datasets by a common ID
      - `assert_frame_equal` to check if two dataframes are equal
      - `np.isclose` to check if two arrays are close
      - tip: look at documentation and test suite of libraries you're using to find the right tools to write tests
  - identify new tests with [Hypothesis](https://hypothesis.readthedocs.io/en/latest/)
    - rather than explicitly state exact objects, hypthesis generates example inputs that follow certain properties you define.
  - test on data: pandera and great expectations
    - why? Testing data is extremely helpful if we will be repeatedly receiving new data with the same structure
    - [pandera](https://pandera.readthedocs.io/en/stable/)
      - define schema for data with `infer_schema`
      - add checks to columns
    - [great expectations](https://greatexpectations.io/)
      - generates a template of tests for data (pandera's `infer_schema` on steroies)
      - data docs feature, communicate data quality issues
  - pytest
    - fixtures: objects commonly used across tests, the utility is so that you don't have to rewrite code to create the same object over and over again
    - arrange-act-assert pattern

Links 🔗

- [Ways I Use Testing as a Data Scientist | Peter Baumgartner](https://www.peterbaumgartner.com/blog/testing-for-data-science/?s=08)
- [Arrange-Act-Assert: A Pattern for Writing Good Tests | Automation Panda](https://automationpanda.com/2020/07/07/arrange-act-assert-a-pattern-for-writing-good-tests/)
- [Ranking YC Companies with a Neural Net | Eric Jang](https://evjang.com/2022/04/02/yc-rank.html)
- [Building a compelling Data Science Portfolio with writing – Weights & Biases](https://wandb.ai/parul_pandey/discussions/Building-a-compelling-Data-Science-Portfolio-with-writing--Vmlldzo4MTA4OTE)

## Day 77: Nov 15, 2022

- check if blocking effective? : F > 1
  - if random blocks model, F > 1 is equivalent to variance component > 0
- cell mean = average Y for one combination of all factors
- marginal mean = avg of cell means, averaged over all other factors
- ls mean (least squares) = either cell mean or marginal mean
- simple effect: difference or linear contrast between two cell mean
- main effect: difference or linear contrast between two marginal mean
- interactions exist when simple effects are not equal
  - in other words: they concern equality of simple effects.
  - ex: does men and women react differently to the treatment?
- effects of unbalanced data
  - type I SS : sequential SS, each term compared to model with "earlier" terms
    - depends on order of terms, when unbalanced data, different order of terms can lead to different type I SS
    - why? contrats that are orthogonal for equal sample size are not orthogonal for unequal sample size
  - type III SS : partial SS, each term compared to model with all other terms except term of interest
    - same for any order of terms
    - same as SS derived using contrasts among cell means
  - note: When a term is the last one in the model, the sequential approach and partial approach will always compare the same pair of models.
- check for missing obsevation
  - check if corrected total error = N - 1
- check for missing cells
  - check if highest interaction term SS has expected df (product of main effect dfs)
- what is ancova?
  - introducing a covariate to a regression model to control for a confounding variable
  - what is covariate? a variable that is measured at the same time as the outcome variable and that might potentially confound the relationship between the independent variable and the dependent variable
  - ex: you want to see if there is a relationship between the number of hours a student studies and their test score. However, you also know that the student's intelligence is a confounding variable. You can control for this by including intelligence as a covariate in your regression model.
  - code example: `model <- lm(score ~ hours + intelligence, data = mydata)` where `score` is the dependent variable, `hours` is the independent variable, and `intelligence` is the covariate.

Links 🔗

- [Using functional analysis to model air pollution data in R | Nicola Rennie](https://nrennie.rbind.io/blog/2022-11-14-using-functional-analysis-to-model-air-pollution-data-in-r/)

## Day 78: Nov 16, 2022

- Techniques to optimize NN model performance
  - parallelization: splitting training data (mini) batches into chucnks and process these smaller chunks in parallel
  - vectorization: replaces costly for loops with operations that apply the same operations to multiple elements (torch.tensor and dot product)
  - loop tiling: change data accessing order in a loop to leverage hardware's memory layout and cache.
  - operator fusion: combine multiple loops into one
  - quantiation :reduce numerical precision (floats -> ints) to speed up computation and lower memory requirements (while maintaining accuracy) (lightnihg.pytorch.callbacks import QuantizationAwareTraining)
- UMAP : Uniform Manifold Approximation and Projection for Dimension Reduction
  - faster than tSNE
  - UMAP overall follows the philosophy of tSNE, but introduces a number of improvements such as another cost function and the absence of normalization of high- and low-dimensional probabilities

Links 🔗

- [rasbt/mlxtend: A library of extension and helper modules for Python's data analysis and machine learning libraries.](https://github.com/rasbt/mlxtend)
- [How Exactly UMAP Works. And why exactly it is better than tSNE | by Nikolay Oskolkov | Towards Data Science](https://towardsdatascience.com/how-exactly-umap-works-13e3040e1668)
- [UMAP paper](https://arxiv.org/pdf/1802.03426.pdf)

## Day 79: Nov 17, 2022

- Get and evaluate startup ideas
  - 4 most common mistakes
    - not solving a real problem (SISP: solution in search of problem ex: AI is cool what can AI be applied to)
    - getting stuck on tarpit ideas (tarpit = set of ideas that have been around forever but have been stuck in tar and don't seem to get anywhere)
      - what causes tarpit ideas? : widespread problem, seems easy to solve, structural reason why it hasn't been solved
    - not evaluating an idea
    - waiting for the perfect idea
  - 10 key question for any idea
    - Do you have a founder/market fit?: pick a good idea for my team!
    - How big is the market?: ex: Coinbase
    - How accute is the problem? ex: Brex
    - Do you have competition?: Most good ideas have competition, but needs a new insight -> Yes I do
    - Do you want this?: Do you know personally who wants this? -> I know someone who wants this, but I am not sure if I want this.
    - Did this recently become possible or necessary? ex: Checkr
    - Are there good proxies for this business? proxies is a large company that does something similar, but not a direct competitor. Rappi example.
    - Is this an idea you'd want to work on for years?: boring idea is okay, and tend to be passionate over time.
    - Is this a scalable business? Software -> check, high skilled human labor.
    - Is this a good idea space? Do you expect is going to have a reasonable hit rate. ex: Fivetran
  - 3 things that make ideas good
    - hard problem ex: Stripe (credit card processing and banks)
    - boring space ex: Gusto (payroll)
    - existing competitors ex: DropBox (19 other competitors which had bad UI, dropbox edge was integration with OS)
  - how to come up with ideas? (3 ways)
    - become an expert on something valuable
    - work at a startup
    - build thigns you find interesting
  - 7 recipe to generate ideas
    - start with what your team is good at : automatic founder market fit (ex: Rezi)
    - start with a problem you've personally encountered (ex: Vetcove)
    - think of things you've personally wish existed (ex: DoorDash)
    - look at things that have changed recently (ex: Covid -> Gathertown)
    - Look for new variants of successful companies (ex: Nuvocargo)
    - talk to people and ask them what problems they have (pick fertile idea space and talk to people in that space ex: AtoB)
    - look for a big industry that seem broken (ripe for disruption)
    - find a cofounder with an idea

Links 🔗

- [How to Get and Evaluate Startup Ideas | Startup School](https://youtu.be/Th8JoIan4dg)

## Day 80: Nov 18, 2022

- Federated learning
  - a general framework that leverages data minimization tactics to enable multiple entities to collaborate in solving a machine learning problem.
  - data minimization: the process of reducing the amount of data that is shared between entities in a federated learning system.
  - secure aggregation: a technique that allows multiple entities to collaboratively compute a function on their local data without revealing their data to each other.
    - Secure aggregation and secure enclaves - combining many local models into an aggregate without revealing the contribution of any user to the server.
    - In the secure aggregation protocol, user devices agree on shared random numbers, teaming up to mask their local models in a way that preserves the aggregated result. The server won’t know how each user modified their model.
  - training a federated model
    - sophisticated models require many iterations of local training and federated averaging
    - local heat map models drift apart after a significant period of local training, and the latest global model’s accuracy might degrade upon merging. Relatively frequent periodic averaging is used to avoid this
  - outliers
    - excluding outliers from training risks reducing accuracy for groups of people less represented in the training pool
    - in practice the server in a federated learning system cannot directly see user training data, which makes detecting outliers in federated learning tricky
  - differential privacy
    - If one user’s participation can significantly affect the model (outlier), then someone observing the final model might be able to determine who participated in training, or even infer their local data
    - Carefully bounding the impact of any possible user contribution and adding random noise to our system can help prevent this, making our training procedure differentially private
    - In practice user models are clipped and noised rather than their raw data, or noise is applied to the combination of many clipped models. Applying the noise centrally tends to be better for model accuracy, however the un-noised models may need to be protected by technologies like trusted aggregators.

Links 🔗

- [How Federated Learning Protects Privacy](https://pair.withgoogle.com/explorables/federated-learning/)
- [Diffie–Hellman key exchange - Wikipedia](https://en.wikipedia.org/wiki/Diffie%E2%80%93Hellman_key_exchange)
- [Differential privacy in (a bit) more detail - Ted is writing things](https://desfontain.es/privacy/differential-privacy-in-more-detail.html)
- [Can a Model Be Differentially Private and Fair?](https://pair.withgoogle.com/explorables/private-and-fair/#:~:text=%E2%9A%AC%20Adding%20random%20noise%20to%20the%20gradient.)
- [TensorFlow Federated](https://www.tensorflow.org/federated)
- [TensorFlow Federated Tutorial Session - YouTube](https://www.youtube.com/watch?v=JBNas6Yd30A)
- [Hidden Technical Debt in Machine Learning Systems](https://proceedings.neurips.cc/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf)

## Day 81: Nov 19, 2022

- all roads lead to rome
  - Ilya Sutskever - "Every Successful (Tech) Company will be an AGI company”
  - improving a product by the same delta involves exponentially harder tech
    - In semiconductor manufacturing, shrinking from 32nm to 14nm process nodes is pretty hard, but going from 14nm to 7nm process nodes is insanely hard, requiring you to solve intermediate problems like creating ultrapure water.
    - Creating a simple Text-to-Speech system for ALS patients was already possible in the 1980s, but improving pronunciation for edge cases and handling inflection naturally took tremendous breakthroughs in deep learning.
    - A decent character-level language model can be trained on a single computer, but shaving a few bits of entropy off conditional character modeling requires metaphorically lighting datacenters on fire.
    - Autonomous highway driving is not too hard, but autonomously driving through all residential roads at a L5 level is considered by many to be AGI-complete.
  - In order to continue adding marginal value to the customer in the coming decades, companies are going to have to get used to solving some really hard problems, eventually converging to the same hard problem: Artificial General Intelligence (AGI)
  - data moats in era of language models
    - Google + Meta -> capture human desire and behaviour and knowledge
    - github -> campture human problem solving
    - gramarly + tiktok + adobe + riot games -> capture human creativity
    - two sigma -> simulate world to predict markets
    - tesla -> capture human decision making in real world
    - all this leads to human thought, creativity, desire, cognition, physical intelligence distilled into a large model.

Links 🔗

- [All Roads Lead to Rome: The Machine Learning Job Market in 2022 | Eric Jang](https://evjang.com/2022/04/25/rome.html)
- [Russell Kaplan on Twitter: "Second order effects of the rise of large language models:" / Twitter](https://twitter.com/russelljkaplan/status/1513128005828165634?s=20&t=QePcsbrR-QVHtegU-3KvrQ)
- [Reality has a surprising amount of detail](http://johnsalvatier.org/blog/2017/reality-has-a-surprising-amount-of-detail)
- [Timing Technology: Lessons From The Media Lab · Gwern.net](https://www.gwern.net/Timing)

## Day 82: Nov 20, 2022

- Cyberpunk 2077 day

Links 🔗

- [Hallucinations re: the rendering of Cyberpunk 2077](https://c0de517e.blogspot.com/search/label/Rendering%20tutorials)

## Day 83: Nov 21, 2022

- how super() works
  - base and derived class, base class is the class that is inherited from, derived class is the class that inherits from another class, it uses super() to call the base class method and then add its own functionality
  - if a derive class looks for a method or attribute in its own class, it will find it and use it, if it doesn't find it, it will look in the base class or any of the base classes of the base class, and so on until the root.
  - this concept is known as next in line, or formally: Method resolution order (MRO) of an object, which defines search order for attribute lookups
- great listening
  - is attentiveness, conveying understanding and showing a positive intention to speaker
  - tips
    - remove any distractions around you, TV, phones, etc. (visible presence of phone made conversations less intimate) and try not to interrupt, if you do interject, ask open-ended question that helps the speaker (how did you that make you feel?)
    - summarize what the speaker said
    - stay present and if you lose focus, and ask to repeat what they said
    - take a moment to formulate your response, gives them time to think about what they said

Links 🔗

- [super, Python's most misunderstood feature. - YouTube](https://www.youtube.com/watch?v=X1PQ7zzltz)
- [4 things all great listeners know - YouTube](https://www.youtube.com/watch?v=i3ku5nx4tMU)

## Day 84: Nov 22, 2022

- build and deploy code (CICD)
  - server (serve front and APIs)
    - horizontal (increase capacity by adding more servers)
    - vertical (increase server specs)
  - storage (database)
  - load balancer (route traffic to server)
  - logging (log events ex: user actions, failed requests, etc.)
  - metrics (track performance ex: response time, memory usage, etc.)
  - alerts (notify when something goes wrong ex: server down, high memory usage, etc.)
- MLE skills
  - computer science fundamentals: data structures, algorithms, object-oriented programming
  - an in-depth understanding of production engineering best practices: monitoring, alerting, unit testing, version control, latency, throughput, scale
  - fundamentals of machine learning and statistics: traditional models, sampling, A/B testing
  - an informed opinion about all the currently popular trends in a machine learning stack so you can select from them (RIP Lambda architecture)
  - an understanding of the developments in the field you specialize in (medical AI, recommendations, search, ML for security, each of which are formulated as different problems and have their own context and vocabulary)
  - YAML
- types of memory (The Programmer’s Brain)
  - As a quick primer, the human brain has several types of memory, short-term, working, and long-term. Short-term memory gathers information temporarily and processes it quickly, like RAM. Long-term memory are things you’ve learned previously and tucked away, like database storage. Working memory takes the information from short-term memory and long-term memory and combines them to synthesize, or process the information and come up with a solution.
- two types of MLE
  - Task MLE, who is responsible for sustaining a specific ML pipeline (or small set of ML pipelines) in production. They are concerned with specific models for business-critical tasks. They are the ones paged when top-line metrics are falling, tasked with “fixing” something. They are the ones who can most likely tell you when a model was last retrained, how it was evaluated, etc.
  - the Platform MLE, who is responsible for helping Task MLEs automate tedious parts of their jobs. Platform MLEs build pipelines (including models) that support multiple Tasks, while Task MLEs solve specific Tasks. It’s analogous to, in the SWE world, building infrastructure versus building software on top of the infrastructure.
- reading AI research papers
  - I’m going to break down the process of reading AI research papers into two pieces: reading wide, and reading deep. When you start learning about a new topic, you typically get more out of reading wide: this means navigating through literature reading small amounts of individual research papers. Our goal when reading wide is to build and improve our mental model of a research topic. Once you have identified key works that you want to understand well in the first step, you will want to read deep: here, you are trying to read individual papers in depth. Both reading wide and deep are necessary and complimentary, especially when you’re getting started.

Links 🔗

- [Anatomy of a Production App - System Design - YouTube](https://www.youtube.com/watch?v=akXP6pC0piE&list=WL&index=7)
- [How I learn machine learning](https://vickiboykis.com/2022/11/10/how-i-learn-machine-learning/)
- [Landscape of Vector Databases - Dmitry Kan - Medium](https://dmitry-kan.medium.com/landscape-of-vector-databases-d241b279f486)
- [MLOps Is a Mess But That's to be Expected - Mihail Eric](https://www.mihaileric.com/posts/mlops-is-a-mess/)
- [Thoughts on ML Engineering After a Year of my PhD | Shreya Shankar](https://www.shreyashankar.com/phd-year-one/)
- [Machine Learning: The High Interest Credit Card of Technical Debt – Google Research](https://research.google/pubs/pub43146/)
- [Harvard CS197: AI Research Experiences](https://www.cs197.seas.harvard.edu/)
- [My Philosophy on Alerting - Google Docs](https://docs.google.com/document/d/199PqyG3UsyXlwieHaqbGiWVa8eMWi8zzAn0YfcApr8Q/edit)

## Day 85: Nov 23, 2022

- DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
  - what: captures the insights that clusters are dense group sof points, The idea is that if a particular point belongs to a cluster, it should be near to lots of other points in that cluster.
  - how:
    - choose two parameters: epsilon and minPts
    - begin picking an arbitrary point in dataset
    - if minPts number of points are within epsilon distance of the point, then it part of a "cluster"
    - expand that cluster by checking all new points, if they too have more than minPoints points within a distance of epsilon, growing the cluster recursively.
    - eventually you run out of new points to add to the cluster and pick a new arbitrary point and repeat the process
    - now you might encounter a point that has fewer than minPts points within epsilon distance, this is a "noise point" not belogning to any cluster.
- cache python
  - `from functools import cache` and `@cache` allows you to cache the result of a function call, so that if you call the function with the same arguments, it will return the cached result instead of recomputing it.
  - variant of cache is `@lru_cache` (least recently used cache) that only saves up to the maxsize most recent calls. It can save time when an expensive or I/O bound function is periodically called with the same arguments.

Links 🔗

- [Visualizing DBSCAN Clustering](https://www.naftaliharris.com/blog/visualizing-dbscan-clustering/)
- [Tracking down the Villains: Outlier Detection at Netflix | by Netflix Technology Blog | Netflix TechBlog](https://netflixtechblog.com/tracking-down-the-villains-outlier-detection-at-netflix-40360b31732)
- [The Single Most Useful Decorator in Python - YouTube](https://www.youtube.com/watch?v=DnKxKFXB4NQ)
- [Principal Component Analysis – Math ∩ Programming](https://jeremykun.com/2012/06/28/principal-component-analysis/)
- [PCA, visualized for human beings | casey.li](https://casey.li/pca/)

## Day 86: Nov 24, 2022

- profiling python code
  - basic profiling with cProfile
  - web based visualization with [snakeviz](https://jiffyclub.github.io/snakeviz/)

```python
import cProfile
import pstats

with cProfile.Profile() as pr:
  my_function()

stats = pstats.Stats(pr)
stats.sort_stats(pstats.SortKey.TIME)
stats.print_stats()
stats.dump_stats(filename="my_stats.prof")
```

```python
import httpx
import asyncio

def my_function():
  # get urls
  async with httpx.AsyncClient() as client:
    tasks = (client.get(url) for url in urls)
    reqs = await asyncio.gather(*tasks)
  # process responses

# to run
asyncio.run(my_function())
```

Links 🔗

- [Diagnose slow Python code. (Feat. async/await) - YouTube](https://www.youtube.com/watch?v=m_a0fN48Alw)
- [jiffyclub/snakeviz: An in-browser Python profile viewer](https://github.com/jiffyclub/snakeviz/)
- [The Python Profilers — Python 3.11.0 documentation](https://docs.python.org/3/library/profile.html)
- [pyutils/line_profiler: Line-by-line profiling for Python](https://github.com/pyutils/line_profiler)

## Day 87: Nov 25, 2022

- why is python slow
  - python is single threaded and runs on a single core
  - why? Global Interpreter Lock (GIL): a mutex that prevents multiple threads from executing Python bytecodes at once. This means that only one thread can be in a state of execution at any given time. This is a problem because it means that if you have a CPU-bound task, you can’t take advantage of multiple cores.
- solution:
  - asyncio
    - cooporative pausing/waiting
    - good for I/O bound operations (network, disk, etc)
  - threading
    - non-cooporative pausing / interrupting
    - good for IO bound
    - good to do long-running operations without blocking
  - multiprocessing
    - each process has its own memory space and GIL, fully utilizing all cores of the CPU
    - `imap_unordered` - returns an iterator that yields results as they become available
    - `imap` - returns an iterator that yields results in the order that they are completed
    - `map` - returns a list of results in the order that they are completed
- multiprocessing pitfalls
  - use multiprocessing where overhead of creating and commucating between them is greater than cost of just doing computation.
    - note: only apply to things that are already taking a long time
  - send or receive across processes boundaries that are not picklable
    - threads share virtual memory so variable in one thread can share memory with another thread
    - processes have their own address space and do not share memory, multiprocessing gets around this by serializing everything with pickle and uses a pipe to send the bytes across the process boundary
  - trying to send too much data
    - instead of passing data, send message like a string that tells process how to create data on its own (ex: filename)
  - using multiprocessing when there's a lot of shared computation between tasks
    - ex: fibonnaaci sequence, huge waste of computing fibonnaci numbers independently as there is a lot of overlap
  - not optimizing chunksize
    - items are split into chunks, and worker grabs an entire chunk of work. bigger chunks allow individual workers to take less trips back to the pool to get more work
    - tradeoff: bigger chunk = copy more items at once across process boundaries (out of memory)
    - consider imap / imap_unordered as it gives results as they come in with an iterator
    - larger chunk size = faster, but more memory, smaller chunk size = slower, but less memory

Links 🔗

- [Unlocking your CPU cores in Python (multiprocessing) - YouTube](https://www.youtube.com/watch?v=X7vBbelRXn0)
- [Python Multiprocessing: The Complete Guide](https://superfastpython.com/multiprocessing-in-python/)
- [multiprocessing — Process-based parallelism — Python 3.11.0 documentation](https://docs.python.org/3/library/multiprocessing.html)
- [concurrent.futures — Launching parallel tasks — Python 3.11.0 documentation](https://docs.python.org/3/library/concurrent.futures.html)

## Day 88: Nov 26, 2022

- 7 deadly sins of speaking
  - gossip: speaking negatively behind someone's back
  - judging: "I'm better than you"
  - negativity: negative viewpoint on things
  - complaining: viral misery
  - excuses: not taking responsibility for actions and blaming others
  - exaggeration: inflating stuff excessively
  - dogmatism: conflating facts with opinions
- foundation of powerful speech (HAIL)
  - Honesty: being clear and straight, but with love and compassion
  - Authenticity: being yourselv and not imitating a non-generic persona. "Standing in your own truth"
  - Integrity: Be your word, do what you say, take responsibility
  - Love: Wishing people well
- Toolbox for speech
  - Register: talk from your chest (not nose or throat), we associate depth with power and authority
  - Timbre: the way voice feels, "Rich, Smooth, Warm, like hot chocolate"
  - Prosody: talking with enthusiasm, rythmic voice, not being monotonic
  - Pace: talk normally and slow right down to emphasize a point
  - Silence: powerful tool to bring attention
  - Pitch: deliver an idea or ask a question and being understood
  - Volumne: quiter the better in bringing attention
- python tips
  - use isinstance instead of == to check for type
  - use if x instead of if len(x) > 0 to check for empty list
  - don't use range(len(a))
  - use time.perf_counter() instead of time.time() for timing
  - use logging instead of print for debugging
  - don't use shell=True in subprocess.run, it's a security risk, use list instead
  - python is compiled to bytecode, which is then ran by the python interpreter

```py
import logging

def my_function():
  logging.debug("debug message")
  logging.info("info message")
  logging.warning("warning message")
  logging.error("error message")
  logging.critical("critical message")

def main()
  level = logging.DEBUG
  fmt = "%(asctime)s %(levelname)s %(message)s"
  logging.basicConfig(level=level, format=fmt)
```

```py
import subprocess

def my_function():
  subprocess.run(["ls", "-l"], capture_output=True)
```

Links 🔗

- [How to speak so that people want to listen](https://www.youtube.com/watch?v=eIho2S0ZahI)
- [25 nooby Python habits you need to ditch - YouTube](https://www.youtube.com/watch?v=qUeud6DvOWI)
- [Logging in Python – Real Python](https://realpython.com/python-logging/)
- [The subprocess Module: Wrapping Programs With Python – Real Python](https://realpython.com/python-subprocess/)

## Day 89: Nov 27, 2022

- Graph Neural Networks
  - why graphs
    - some data are naturally described as graphs, ex: social networks, chemical compounds, protein interactions, etc
    - in deep learning, we assume a structure on data. images (grid of pixels), text (sequence of words), etc. The assumption of that the data has spacial locality is what most of the methods we use are built on. With graphs, the goal is to go beyond euclidian geometry.
  - what is a graph
    - G = (V, E)
    - a set of vertices (nodes) and edges
    - undirected or directed
    - each nodes has a feature vector / embedding of some size
    - edges have a feature vector that describes the relationship between the nodes
    - ex: each node is a molecule, feature vector is the chemical properties of the molecule, edge is the bond between two molecules
  - common graph tasks
    - node classification: classify each node (ex: fraud detection)
    - graph classification: entire graph classification (ex: is it a toxic molecule)
    - node clustering: identify nodes that are similar to each other (clustering)
    - link prediction: predict the existence of an edge between two nodes (ex: user liking a particular movie)
    - influence maximization: find particuclar node that has the most influence on the graph (ex: finding the most influential person in a social network)
  - representation of graph
    - adjacency matrix: matrix of size n x n where n is the number of nodes. each entry is 1 if there is an edge between the two nodes, 0 otherwise. can be sparse if there are few edges
    - feature matrix (X): matrix of size n x d where n is the number of nodes and d is the dimension of the feature vector. each row is the feature vector of the node
  - how a GNN works?
    - in convolution: sliding a filter/kernel with particular weights, and does dot product with the underlying pixel values to get a new value
    - in graphs: local neighborhood for each nodes, perform computation on the neighborhood and update the node
  - information propagation
    - information propagates throughout the network if we stack a bunch of layers, layers == shortes path between nodes
  - key properties
    - permutation invariance: the order of input nodes does not affect the output of network, a desirable property as it allows the network to learn the underlying structure of the graph
    - permutation equivariance: permutation applied to input nodes should be the same as applying the same permutation to the output nodes
  - message passing computation
    - GNN layer 1
      - (1) message : compute a message to send to the output node (done in parallel for all nodes)
      - (2) aggregation : aggregate so order of nodes don't matter (ex: sum, mean, max, etc)
      - (3) update : update the node embedding
  - GNN variants
    - convolution: sum of messages from neighbors normalized by the degree of the node, multiply by a weight matrix, adding the node embedding, and applying a non-linearity function
    - attention: similar process, but adds an importance score to the nodes

Links 🔗

- [Graph Neural Networks: A gentle introduction](https://www.youtube.com/watch?v=xFMhLp52qKI&t=6)
- [Stanford CS224W: Machine Learning with Graphs](https://www.youtube.com/playlist?list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn)
- [A Gentle Introduction to Graph Neural Networks](https://distill.pub/2021/gnn-intro/)
- [Understanding Convolutions on Graphs](https://distill.pub/2021/understanding-gnns/)
- [Graph Neural Networks](https://www.youtube.com/playlist?list=PLV8yxwGOxvvoNkzPfCx2i8an--Tkt7O8Z)
- [Intro to graph neural networks (ML Tech Talks)](https://www.youtube.com/watch?v=8owQBFAHw7E)
- [Graph Neural Nets | AI Epiphany](https://www.youtube.com/playlist?list=PLBoQnSflObckArGNhOcNg7lQG_f0ZlHF5)

## Day 90: Nov 28, 2022

- Python Generators
  - used for lazy evaluation
  - ex: range in Python, file processing
  - generator pipelines
    - define generator comprehensions line by line
    - at teh point of calling the generator, the generator is not evaluated, it is only evaluated when the generator is iterated over
  - advanced usage: send, throw, close
    - yield is not just a statement, it also returns a value
    - generators are bidirectional pipelines, it can yield value to caller, its caller can also send back values to the generator
    - submit tasks to worker, something drives worker, worker decides how task is scheduled and when to call the function and do the work, this is like async (coroutines are defined with generators)
  - `yield from` - allows you to yield from another generator
    - ex: `yield from range(10)` is the same as `for i in range(10): yield i`
    - true purpose: passing values from caller to subgenerator
    - caller <-> quiet worker <-> worker

```py
def worker(f):
  tasks = collections.deque()
  value = None
  while True:
    batch = yield value
    value = None
    if batch is not None:
      tasks.extend(batch)
    else:
      if tasks:
        args = tasks.popleft()
        value = f(*args)

def example_worker():
  w = worker(str)
  w.send(None)
  w.send[(1,), (2,), (3,)]
  w.throw(ValueError) # acts as if the worker threw an exception
  print(next(w))
  print(next(w))
  w.send[(4,), (5,)]
  w.close() # throws special generator exit exception

def quiet_worker(f):
  while True:
    w = worker(f)
    try:
      return_of_subgen = yield from w # pass messages from caller directly to worker
    except Exception as exc:
      print(f"ignoring {exc.__class__.__name__}")
```

Links 🔗

- [Python Generators](https://www.youtube.com/watch?v=tmeKsb2Fras&t=737s)
- [In practice, what are the main uses for the "yield from" syntax in Python 3.3?](https://stackoverflow.com/questions/9708902/in-practice-what-are-the-main-uses-for-the-yield-from-syntax-in-python-3-3)
- [How to Use Generators and yield in Python – Real Python](https://realpython.com/introduction-to-python-generators/)

## Day 91: Nov 30, 2022

- `git reflog` to see the history of your git commits, find the one before you broke everything
- `git reset HEAD@{index}` to go back to that commit
- `git commit --amend --no-edit` to make changes to previous commit
- `git commit --amend` to change last commit message
- `git revert [saved hash]` to undo a commit
- `git checkout [saved hash] -- path/to/file` to undo changes to a file

Links 🔗

- [Oh Shit, Git!?!](https://ohshitgit.com/)
- [SQL Indexing and Tuning e-Book for developers: Use The Index, Luke covers Oracle, MySQL, PostgreSQL, SQL Server, ...](https://use-the-index-luke.com/)

## Day 92: Nov 30, 2022

- useful tips I got working on my GPT3 side project
  - get project directory
    - `project_dir = Path(__file__).resolve().parents[2]`
  - find dot env file
    - `load_dotenv(find_dotenv())`
- article recommender system
  - problem: how do we find good follow-up articles?
  - how? using language models to generate embeddings as a basis for similarity search
  - implementation and training
    - effective method for fine tuning is training via triplet loss
    - loss function: `max⁡{d(a, p)−d(a, n) + 𝛼, 0}`
      - loss is 0 if anchor a and positive example p are closest and grows in proportion as negative example n gets closer
      - 𝛼 is slack variable declaring a soft margin - if the difference in distance is within this margin we always penalize this model
      - d(a, p) is commonly the cosine distance
      - anchor: source article, positive: follow-up article, negative: random article
    - model
      - pre-trained BERT model with average pooling custom head to reduce embedding vector size
    - evaluation
      - 5% holdout set of articles, non-customized BERT vs custom BERT, similarity bewteen anchor and recommendation is much higher than random in custom.
      - visualize embeddings with 3D PCA projection to identify clusters
    - deployment
      - AWS Sagemaker endpoint
      - trigger lambda function that extracts dat and calls endpoint
      - generated embeddings stored in DynamoDB and OpenSearch cluster
      - for every request, look up embedding in table, OpenSearch supports fast KNN search
      - query cluster for recommendations, re-rank them including age and popularity, send top 3 candidates to users
- triplet loss:
  - Triplet loss is a type of learning algorithm used in machine learning. It is called "triplet loss" because it uses three things to help a computer learn: a positive example, a negative example, and a "anchor" point.
  - Imagine you have a computer that is trying to learn what a cat looks like. You show the computer a picture of a cat and say, "This is a cat." This is the positive example. Then, you show the computer a picture of a dog and say, "This is not a cat." This is the negative example. Finally, you show the computer a picture of a cat again and say, "This is also a cat." This is the anchor point.
  - The computer uses these three examples to learn what a cat looks like. It compares the positive and negative examples and tries to figure out what makes them different. Then, it compares the anchor point to the positive example and tries to make sure they are similar and push it closer to the anchor than the negative example.
- pooling
  - The purpose of a custom pooling head is to allow the model to output a fixed-size representation (i.e. a "pooled" representation) of the input text that can be used for downstream tasks, such as classification or regression.
  - In BERT, the pre-trained model outputs a sequence of hidden states for each input token. These hidden states capture the contextual information about the input, but they are not fixed-size vectors, so they cannot be used directly as input to a downstream task. A custom pooling head solves this problem by aggregating the hidden states in some way (e.g. by taking the mean or maximum value) to produce a fixed-size representation of the input text. This fixed-size representation can then be fed into a downstream task-specific model, such as a classification model or a regression model.
  - how to choose?
    - The type of downstream task: Different downstream tasks (e.g. classification, regression, semantic similarity) may require different types of pooled representations of the input text. For example, a classification task may benefit from a pooling head that captures the global context of the input text, while a regression task may benefit from a pooling head that captures the local context of the input text.
    - The size of the input text: If the input text is very long, a pooling head that captures the global context of the input text may be more appropriate, as it will be able to produce a fixed-size representation that summarizes the entire input text. If the input text is very short, a pooling head that captures the local context of the input text may be more appropriate, as it will be able to produce a fixed-size representation that preserves the fine-grained details of the input text.

Links 🔗

- [Better article recommendations with triplet fine-tuning](https://medium.com/s%C3%BCddeutsche-zeitung-digitale-medien/better-article-recommendations-with-triplet-fine-tuning-6b52a587b85f)
- [NLP: Everything about Embeddings. Numerical representations are a… | by Mohammed Terry-Jack | Medium](https://medium.com/@b.terryjack/nlp-everything-about-word-embeddings-9ea21f51ccfe)
- [The Illustrated Word2vec – Jay Alammar – Visualizing machine learning one concept at a time.](https://jalammar.github.io/illustrated-word2vec/)
- [Embedding projector - visualization of high-dimensional data](http://projector.tensorflow.org/)
- [7.5. Pooling — Dive into Deep Learning 1.0.0-alpha1.post0 documentation](https://d2l.ai/chapter_convolutional-neural-networks/pooling.html)
- [Adding Custom Layers on Top of a Hugging Face Model | by Raj Sangani | Towards Data Science](https://towardsdatascience.com/adding-custom-layers-on-top-of-a-hugging-face-model-f1ccdfc257bd)
- [ENC2045 Computational Linguistics — ENC2045 Computational Linguistics](https://alvinntnu.github.io/NTNU_ENC2045_LECTURES/intro.html)

## Day 93: Dec 1, 2022

- worked on gpt3-blog-title
- docker stuff
  - `docker build -t [image name] .` to build docker image
  - `docker run -p 5000:5000 [image name]` to run docker image
  - `docker builder prune` to remove unused images

Links 🔗

- [benthecoder/gpt3-blog-title: Using GPT-3 to help me get more claps on medium.com](https://github.com/benthecoder/gpt3-blog-title)
- [Deploy Streamlit using Docker - Streamlit Docs](https://docs.streamlit.io/knowledge-base/tutorials/deploy/docker)

## Day 94: Dec 2, 2022

- What's a CDN?
  - Content Delivery Network (CDN) was originally developed to speed up the delivery of static content such as HTML pages, images, and JavaScript files. Nowadays it is used whenever HTTP traffic is served
  - The goal of a CDN is to deliver content to end-users with high availability and high performance.
  - It is a network of servers that are deployed in multiple data centers around the world, known as PoPs (Points of Presence).
  - A server inside a PoP is called an Edge Server.
  - Two common CDNs
    - DNS-based routing
      - each PoP has it's own IP address, and the DNS server will return the IP address of the PoP that is closest to the end-user
    - anycast
      - all PoPs have the same IP address, and the end-user's request will be routed to the PoP that is closest to the end-user
  - Each edge server acts as a reverse proxy with huge content cache
    - greatly reduced the load and latency of the origin server
  - modern CDN also transforms static content into more optimized format
    - minify JS bundles on the fly
    - transform image files to WEBP and AVIF format
  - TLS handshakes are expensive, so TLS termination is done at the edge server, which reduces latency for user to establish and encrypted connection
    - This is why modern apps send dynamic uncacheable content directly to the edge server, and let the edge server handle the TLS termination
  - security
    - all modern CDNs have huge network capacity at the edge, which prevents DDoS attacks by having a much larger network than attackers
    - This is especially effective in Anycast which diffuses the attack across all PoPs
  - improves availability
    - by nature is highly distributed by having copies of contents available in multiple PoPs, it can handle many more hardware failures than origin server.
- What is gRPC?
  - RPC stands for Remote Procedure Call, which enables one machine to call a function on another machine as if it was a local function call
  - gRPC is a popular implementation of RPC
  - why is it popular?
    - thriving developer ecosystem, with support for many languages and the core of this support is the use of Protocol Buffers (protobuf)
      - What is protobuf?
        - language-agnostic and platform-agnostic mechanism for encoding structured data
      - why not JSON?
        - protobuf supports strongly typed schema data, which means that the data is validated before it is sent over the wire
        - provide broad tooling support to turn schema defined into data access classes for many languages
      - gRPC is defined in a .proto file by specifing RPC method parameters and return types.
      - same tooling is used to generate client and server code from proto file, which is used to make RPC calls (client) and fulfill RPC requests (server)
    - highly performant out-of-the-box
      - 1. uses efficieny binary encoding (5x faster than JSON)
      - 2. Built on http/2, which allows multiply streams of data to be sent over a single long-lived TCP connection. This means it can handle many concurrent RPC calls over a small number of TCP connections
  - why isn't this popular in web clients and servers?
    - gRPC relies on lower-level access to http/2 primitives, it's possible with gRPC-web which makes gRPC calls with a proxy. However, the feature set is not fully compatible
  - when to use it?
    - inter-service communication mechanism of choice between microservices in data centers
    - in native mobile clients

Links

- [What Is A CDN? How Does It Work? - YouTube](https://www.youtube.com/watch?v=RI9np1LWzqw)
- [What is RPC? gRPC Introduction. - YouTube](https://www.youtube.com/watch?v=gnchfOojMk4)

## Day 95: Dec 3, 2022

- Tips for Writing Functions
  - (1) do one thing and do it well
    - take out low-level things and put into another function
  - (2) separate commands from queries
    - either retrieve information or perform an action (command-query separation principle)
  - (3) Only request information you actually need
    - passing in arguments that the function needs instead of an entire class
  - (4) keep number of parameters minimal
    - good indication of how much the function is supposed to do
    - provide default values if possible
    - introduce abstraction that represent arguments
      - Protocol class for function arguments (Card protocol for Customer class)
      - dataclasses to represent subobjects (respresent data better)
  - (5) don't create and use object in same function
    - forces you to use a specific implementation (ex: Stripe class for payment, can't use other payment methods)
    - either pass in class as argument or introduce abstraction by creating a protocol (ex: Payment protocol)
  - (6) don't use flag arguments
    - what are flag arguments? a function argument of a boolean type that alters function behaviour
    - using this is example of a code smell
    - should be two functions: one for the case where the flag is true, and one for the case where the flag is false
  - (7) remember functions are objects
    - `from typing import Callable`
      - allow you to pass functions as arguments instead of having to instatiate a class
    - `from functools import partial`
      - create a subfunction that is a partial application of another function (ex: `dbl = partial(multiply, 2)`)
  - Bonus tip
    - check functions with "and" in name, they should be split up
    - choose good argument names `publish_info_to_library(lib)` -> `publish_info_to(library)`
    - should be actions (verbs) and arguments should be nouns
    - use the same vocabulary for arguments and functions
    - use naming scheme that language describe (snake_case, camelCase, PascalCase)
    -
- Force use of keyword arguments
  - `def my_function(*, name: str, age: int):` - forces any function call to use keywords

Links 🔗

- [The Ultimate Guide to Writing Functions - YouTube](https://www.youtube.com/watch?v=yatgY4NpZXE)
- [Building Implicit Interfaces in Python with Protocol Classes](https://andrewbrookins.com/technology/building-implicit-interfaces-in-python-with-protocol-classes/)
- [Python Protocol](https://www.pythontutorial.net/python-oop/python-protocol/)

## Day 96: Dec 4, 2022

- docker stuff
  - copy requirements and install first then copy app so that installation of requirements (which takes longer) is cached
  - what is docker compose for?
    - when deploying code to cloud, you need to build docker images just like the way before and tell kubernetes to update the running image to the next version (done with CI/CD like github actions)
    - doing it locally is not great, every time you change code -> stop server -> rebuild docker image -> restart container
    - 2 main features `docker-compose.yml`
      - (1) custom run command that restarts automatically when file has been change
      - (2) sync folder in your local machine (volume) to a folder running inside the container
  - use multi-stage builds, docker images' size gets really blown out when you create many layers on them
  - not use the latest image unless you're running some testing for building docker images, I've met some situations when something changed between let's say python3.6->3.9 and the docker image would be non functional
  - look into docker-slim after the two-stage images, Your image is probably ~800-900 MB, two stage build with alpine-python or python:slim would make it go to ~100-150MB and after docker slim you could be left with at most 50MB.
  - building with Kaniko (tool from google) instead of docker build, it's faster, produces slightly lighter images and has better caching at least in my experience

```py
FROM python:3.9-alpine

# set the working directory
WORKDIR /app

# install dependencies
COPY ./requirements.txt /app
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# copy the scripts to the folder
COPY . /app

# start the server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]

```

Links 🔗

- [How To Use Docker To Make Local Development A Breeze - YouTube](https://www.youtube.com/watch?v=zkMRWDQV4Tg)

## Day 97: Dec 5, 2022

- Storing passwords
  - Not to do: storing in plain text
  - Right way: Open Web Application Security Project (OWASP) guidelines
    - (1) use modern hashing algorithm
      - hasing is a one way function, impossible to decrypt a hash to obtain original password
      - faster functions like md5 and sha1 are not secure enough, they can be cracked with brute force attacks
      - bcrypt -> slow functions
    - (2) salt the passwords, salt is unique randomly generated string that is added to each password
      - hash(password + salt)
      - why salt? storing password as one-way hash is not sufficient, attacker can defeat it with pre-computation attacks like rainbow tables and database-based lookups to find the password. Adding a salt makes the hash unique to each password
    - process
      - password + salt -> hash -> store `id, hash, salt` in database
      - user login -> lookup user in database -> hash password + salt -> compare with hash in database
  - additional stuff
    - In addition to a per-password salt you can also add a random, application-specific "pepper". It's hashed along with the password and the salt, but unlike the salt the pepper is not stored in the DB. A dump is useless without the pepper and an attacker would often need multiple vectors to obtain both.

Links 🔗

- [System Design: How to store passwords in the database? - YouTube](https://youtu.be/zt8Cocdy15c)
- [Hashing in Action: Understanding bcrypt](https://auth0.com/blog/hashing-in-action-understanding-bcrypt/)
- [Lecture24.pdf](https://engineering.purdue.edu/kak/compsec/NewLectures/Lecture24.pdf)

## Day 98: Dec 6, 2022

- Science-Based Tools For Increasing Happiness
  - describing happiness
    - Language is not always a sufficient tool for describing states of the brain and body
    - No single neurotransmitter or neuromodulator is responsible for the state of happiness – but it is true that people with lower levels of dopamine and serotonin at baseline self-report lower levels of happiness
  - two types of happiness
    - natural: obtained through gift or effort
    - synthetic (self-induced): focusing on leveraging opportunities and making choices that enhances happiness
      - about managing the emotional and reward system of the brain –the anticipation of something can make you as happy as the thing itself
      - We need to make an effort toward being happy: focus on the things that bring meaning and engage in things that bring meaning – adjust home and work environment so it’s pleasant for you
  - trends in happiness
    - Takeaways from the longest longitudinal study on happiness (75 years): (1) money doesn’t create happiness – past a certain level of income, happiness doesn’t scale with money (though it does buffer stress) (2) total amount of time spent working does not create happiness
    - The trajectory of happiness across one’s lifespan is a U-shaped curve: people in 20s tend to be happiest until responsibilities pick up in their 30s, 40s, and 50s; happiness picks up again in their 60s and 70s when demands reduce
    - People age 25 or older tend to be less happy on their birthdays because we get a snapshot of where our lives are compared to peers, or what we’ve accomplished relative to age
  - tips
    - (1) prosocial spending: giving increases happiness for the giver (money, effort, and time)
    - (2) staying focused: focusing on the current activity leads to higher self-reported happiness than mind wandering (meditation can enhance ability to focus)
    - (3) quality social connection: relationships induce happiness, we have a whole area of the brain dedicated to facial recognition
      - two forms: presence and mutual eye contact and nonsexual physical contact
    - (4) pet: even seeing dogs stimulates happiness (foster dogs, visit shelters, walk dogs)
  - sunlight = happiness - viewing direct morning sunlight within 30-60 min helps you fall asleep and stay asleep at night, and optimize cortisol and adenosine levels -

Links 🔗

- [Science-Based Tools for Increasing Happiness | Huberman Lab Podcast #98 - YouTube](https://www.youtube.com/watch?v=LTGGyQS1fZE)
- [Spending Money on Others Promotes Happiness | Science](https://www.science.org/doi/10.1126/science.1150952)
- [A Wandering Mind Is an Unhappy Mind | Science](https://www.science.org/doi/10.1126/science.1192439)
- [The Molecule of More: How a Single Chemical in Your Brain Drives Love, Sex, and Creativity—and Will Determine the Fate of the Human Race by Daniel Z. Lieberman](https://www.goodreads.com/book/show/38728977-the-molecule-of-more)

## Day 99: Dec 7, 2022

- why is kafka fast
  - what does fast mean? latency or throughput, compared to what?
  - kafka is optimized for high throughput, designed to move large number of records in a short amount of time
    - think of it like a large pipe moving liquid, the bigger the diameter of the pipe, the larger the volume of liquid
  - kafka is fast in the sense that it can move a lot of data efficiently
  - kafka's design choices
    - (1) sequential I/O
      - two types of access patterns
        - random : hard drives, takes time to physically move the arm to different locations on the magnetic disk, makes it slow
        - sequential : arm moves one after the other, much faster to read and write blocks of data since it doesn't have to jump around
      - uses append-only log as its primary data structure (sequential access pattern)
      - metrics
        - sequential: 100MB/s
        - random: 100KB/s
      - HDDs 1/3 of price but 3x storage capacity of SSDs
    - (2) zero copy principle
      - modern unix systems use zero copy to read data from disk to memory
      - without zero copy
        - disc -> OS buffer -> kafka application buffer -> socket buffer -> NIC buffer -> consumer
      - with zero copy
        - disc -> OS Buffer -> (direct copy) NIC buffer -> consumer
        - this copy is done with DMA (direct memory access) which is a hardware feature that allows the NIC to directly access the memory of the OS

Links 🔗

- [System Design: Why is Kafka fast? - YouTube](https://www.youtube.com/watch?v=UNUz1-msbOM)
- [The Log: What every software engineer should know about real-time data's unifying abstraction | LinkedIn Engineering](https://engineering.linkedin.com/distributed-systems/log-what-every-software-engineer-should-know-about-real-time-datas-unifying)
- [Confluent Developer: Your Apache Kafka® Journey begins here](https://developer.confluent.io/)
- [Kafka as a Platform: the Ecosystem from the Ground Up](https://talks.rmoff.net/8OCgKp/kafka-as-a-platform-the-ecosystem-from-the-ground-up)
- [aiven/python-notebooks-for-apache-kafka: A Series of Notebooks on how to start with Kafka and Python](https://github.com/aiven/python-notebooks-for-apache-kafka)

## Day 100: Dec 8, 2022

- The science of falling in love
  - first stage of love: infatuation (passionate love) (few months)
    - show increased activation in Ventral Tegmental Area (VTA)
      - VTA is reward processing and motivation hub of the brain, firing when you do things like eating a sweet treat or in extreme cases taking drug of abuse
      - activation releases the "feel good" neurotransmitter dopamine, teaching your brain to repeat behaviours in anticipation of reward
      - this increase VTA activity is why love is euphoric and what draws you to your new partner
    - hard to see faults in new partner
      - this haze is love's influence on higher cortical brain regions
      - newly infatuated individuals show decreased activity in brains cognitive center, the prefrontal cortex (for critical thoughts and pass judgement)
  - second stage : attatchment (compassionate love) (longer lasting)
    - feel relaxed and committed to partner due to oxytocin and vasopressin
      - known as pair-bonding hormones - signal trust, social support and attatchment, also helsp bond families and friendships
    - oxytocin inhibit release of stress hormones, which is why you feel relaxed with loved ones
    - love's suspension of judgement fades, replaced by honest understanding and deeper connection
    - problems in relationship becomes more evident
  - heartbrakes
    - distress of breakup activates insular cortex
      - regions that processes pain (physical and social)
      - as time passes, daydream / crave contact with partner, drive to reach out like extreme hunger or thirst
    - increased activity in VTA
    - activates your body's alarm system, the stress axis, leaving you shaken and restless
    - higher cortical regions which oversee reasoning and impulse control pump brakes on the distress and craving signaling

Links 🔗

- [The science of falling in love - Shannon Odell - YouTube](https://www.youtube.com/watch?v=f_OPjYQovAE)
- [The Anatomy Of Love](https://theanatomyoflove.com/)
- [Love, Actually: The science behind lust, attraction, and companionship - Science in the News](https://sitn.hms.harvard.edu/flash/2017/love-actually-science-behind-lust-attraction-companionship/)

## Day 101: Dec 9, 2022

- Rust in 10 minutes
  - `let` introduces variable binding - `let x = 5;`
  - types can be annotated `let x : i32 = 5;`
    - python is dynamically typed, quietly promote length of integer at runtime, inefficient when you know how big ur numbers will be
    - use i32 for everything
  - you can't access unitialised variables
    - doing `let x;` and passing it into a function without giving it a variable
    - doing this in C gives undefined behaviour
  - use `_` to throwaway something - `let _ = get_thing()`
  - Rust has tuples `let pair = ('a', 17);`
    - types are nearly always inferred
    - can be destructured `let (_, right) = slice.split_at(middle)`
  - semi-colon marks end of statement
    - can span multiple lines
  - `fn` declares a function
    - void functions return an empty tuple `()` pronounced as a "unit"
    - use -> to indicate return types `fn my_func() -> i32 {4}`
  - brackets `{}` declares a block
    - like immediate function in JS
    - code in block ony lives in that block, does not affect outer scope variables
    - `let x = 42;` is same as `let x = {42};`
    - a block can have multiple statements, the final expression is the tail
    - emitting semi-colon at the end of a function is the same as returning
    - if conditionals + match are also expression
  - dots are used to access fields or call method
  - `::` operates on namespaces
    - `crate::file::function`
    - `use` brings a namespace into scope, Rust has strict scoping rules, if you don't see it in your source code, it's not available
    - types are namespaces too, and methods can be called as regular functions `str::len("hello")`is same as `"hello".len()`
  - `struct` - lightweight new types that encapsulate the valid states of your system
    - you can declare methods on types
    - `struct Number`, then do `impl Number { fn is_positive(&self) {}}`
  - `match` arms are patterns
    - it has to be exhaustive, at least one arm needs to match and `_` is a catch-all
  - variable bindings are immutable by default
    - like Haskell, C like languages add immutanbility with const
    - `mut` makes vairable mutable
  - functions can be generic - `fn foo<T>(arg T) { ... }`
    - think of them like a template string
    - structs can be generic too
    - standard library type `Vec` is a generic too
  - Macros - `name!()`, `name![]`, `name!{}` are macros
    - `!` indicates a macro
    - `vec![]` is a macro that expands to `Vec::new()`
    - `panic!` is a macro that prints a message and exits the program
      - `Option` type can contain nothing, if `unwrap()` is called, it panics
    - macros expands to regular code
  - Enums - two variants
    - `Option` : `Some(T)` or `None`
    - `Result` : it can return a value or an error
      - also panics when unwrapped containing an error
      - Rust has a pattern of errors as values which keeps us in the functional world where other languages would have exceptions that breaks us out
      - `.expect()` for custom error message
      - handling errors
        - match and handle error
        - `if let` to safely destructure the inner value if it is ok
        - bubble up error, returning it to caller
      - this pattern of unwrapping a value inside a Result if it's okay is so common that Rust has dedicated syntax for it - `?` operator
  - iterators
    - computation only happens when it is called (computed lazily)
    - `let natural_numbers = 0..;` goes up to std::i32::MAX (2,147,483,647)
    - the most basic iterators are ranges
      - `(0..).contains(&100);` (bottom)
      - `(..=20).contains(&20);` (top)
      - `(3..6).contains(&4);` (exactly)
    - anything that is iterable can be used in a for loop
      - used with Vec, a slice `(&[...])`, or an actual iterator (string is iterable)
- what is `&` in Rust
  - passing reference of a variable instead of its value toa function
  - reference is a pointer that leads to that variable, Rust uses the concept of ownership to manage memory
  - when a function takes in a reference as a parameter, it is called borrowing
  - ex: if you have a function that takes in a string, it will take ownership of that string, and you can't use it anymore, but if you pass in a reference, you can still use it later on

Links 🔗

- [Rust for the impatient (Learn Rust in 10 minutes) [RUST-4] - YouTube](https://www.youtube.com/watch?v=br3GIIQeefY&list=PLZaoyhMXgBzoM9bfb5pyUOT3zjnaDdSEP&index=6)
- [A half-hour to learn Rust](https://fasterthanli.me/articles/a-half-hour-to-learn-rust)
- [Rust articles](https://fasterthanli.me/tags/rust)
- [The Rust Book](https://doc.rust-lang.org/stable/book/title-page.html)
- [variables - What does the ampersand (&) before `self` mean in Rust? - Stack Overflow](https://stackoverflow.com/questions/31908636/what-does-the-ampersand-before-self-mean-in-rust/31908687#31908687)
- [joaocarvalhoopen/How_to_learn_modern_Rust: A guide to the adventurer.](https://github.com/joaocarvalhoopen/How_to_learn_modern_Rust)
- [Rust By Example](https://doc.rust-lang.org/rust-by-example/index.html)
- [Rust Application Books - The Little Book of Rust Books](https://lborb.github.io/book/applications.html)
- [rust-lang/rustlings: Small exercises to get you used to reading and writing Rust code!](https://github.com/rust-lang/rustlings)
- [Read Rust](https://readrust.net/)
- [This Week in Rust](https://this-week-in-rust.org/)

## Day 102: Dec 10, 2022

- Living a long and awesome life
  - Fats
    - eat a higher proportion of (good) fat -> leaner and healthier
    - types of fat
      - monounsaturated and polyunsaturated fats (GOOD)
        - mono : poultry, fish, avocado, olives, or nuts
        - poly: sesame oil
        - Omega 3 fatty acids. Salmon, trout, sardines, walnuts, flaxseed or supplements are great sources
      - saturated fat (NOT GOOD)
        - animal product like meat, dairy and eggs,
      - artificial trans fats (BAD)
        - margarine, most doughnuts, pastries and cookies, powdered creamer, vegetable shortening, fats in deep fried foods
        - up the chance of heart disease, diabetes, dementia
  - Carbs
    - eat more whole grains, reduce less refined carbs and sugar
    - refined carbs -> increased risk of heart attack, stroke and diabetes
    - GOOD: Whole wheat, barley, wheat berries, quinoa, oats, brown rice, beans, and foods made with them
    - pretty GOOD: Oatmeal, sweet potatoes, and some whole-grain crackers.
    - BAD: Sugar-sweetened soda and fruit juice, white rice, white bread, fries from France, baked potatoes, and pizza.
  - Protein
    - get protein from get more of your protein from plants, beans, nuts, fish, and poultry
    - An ounce of nut = 8 grams of protein, which is the same as a glass of milk
    - too much red meat -> processed meat is “carcinogenic to humans,” and red meat is “probably carcinogenic.”
  - Fruits and Veggies
    - eating lots of fruit and vegetables with “lower risk of dying from any cause.”
    - Different fruits provided protection from different cancers
    - Berries were shockingly beneficial. Potatoes and corn act more like refined carbs in your body
    - Limit fruit juice and smoothies (high sugar and calories)
  - Beverages
    - Water >>>>
    - Coffee: lowers incidence of diabetes and Parkinson’s disease, less liver cancer, reduced kidney stones and gallstones, and lower overall mortality, reduce depression
    - Tea: no clear evidence of quantum healing effects
    - alcohol: raises good cholesterol and reduces the chance of heart attack and stroke (younger people)
    - Vitamins: the basics (especially vitamins B6 and B12, folic acid, vitamin D, and beta-carotene
  - Sum up
    - whole grain bread > white bread
    - brown rice > white rice / potatoes
    - olive oil > better
    - peanut butter > cheese
    - nuts > cheese / sweets
    - beans, soy, fish, poultry > red/processed meat
    - yoghurt + fruits > ice cream

Links 🔗

- [This Is How To Have A Long Awesome Life - Barking Up The Wrong Tree](https://bakadesuyo.com/2022/10/nutrition/)
- [Seven Countries Study - Wikipedia](https://en.wikipedia.org/wiki/Seven_Countries_Study)

## Day 103: Dec 11, 2022

- Building habits
  - you're not making a decision, the brain doesn't have to use any energy to make that decision, happens without will power and any effort
  - invisible architecture of everyday life 40% of life are made of habits
  - habit loop
    - (1) queue : trigger
    - (2) behaviour : routine
    - (3) reward : why the habbit happens
  - queue + reward intertwine -> craving that drives the behaviour
  - world around us is desgined to distract us, we make decisions as a function of the environment
  - committment effect
    - sticking to business plan or carreer or relationship long after it's clear that it's descructive because it's become a part of our identity and we don't want past investment to go to waste
  - look at problems as an outsider
    - you're trapped within your own perspective and emotions and feelings, but giving advice to someone else, we're not trapped, and can give advice that are more forward looking
  - thinking about bad habits
    - "what do I want long term?", "what's really most important to me?"
  - systems > goals
    - life is a series of goals (unsuccesful -> hitting the goal -> not satisfied -> next goal ).
    - Goals are broken processes, they don't tell you how to get to where you're going
    - setting systems allow you to achieve something consistently (ex: an hour a day of writing)
    - Systems are geared towards psychological well-being, it gives a positive feedback we seek
  - temptation bundling
    - pairing "want" activity with "should" activity
  - treat is not reward
    - something you get because you want it, it sounds selfish but it's important to help us get self command. When we give more to ourselves, we can ask more from ourselves

Links 🔗

- [Habits: How to be successful every day | Dan Ariely, Gretchen Rubin & more](https://www.youtube.com/watch?v=Ogc8JUn-F5I)

## Day 104: Dec 12, 2022

- 40 useful concepts

  - Baader-Meinhof Phenomenon: notice something new, start seeing it more often, leading us to confuse attention with reality itself
  - Ostrich Effect: we avoid info that we fear will cause us stress (bills and work emails), this is counterproductive and only prolongs problems
  - Nobel Disease: idolizing people -> inflate their egos and afflict them with hubris to talk abou things they know little about (By celebrating people for their intelligence, we make them stupid.)
  - Warnock's Dilemma: content that provoke people gets more engagement, incetivizing content creators to be provocative
  - Google Scholar Effect: top search results dominate a topic, gatekeeping other viewpoints
  - Paradox of Unanimity: the more people agree, hte less likely they are thinking for themselves, beware of consensus
  - Epistemic Humility: Try to be less wrong instead of trying to be right, always start with - you don't know enough
  - Mimetic desire: watching people want a thing makes us want it too, craving is contagious
  - Overblown Implications effect: People don't judge you by a single success or failure
  - Ellsberg Paradox: People prefer a clear risk over an unclear one, even if it's no safer (explains market volatility)
  - Veblen Goods: we attatch value to things that are hard to get
  - Peter principle: people in hierarchies are promoted until they suck at their jobs, at which they remain where they are (the world is filled with people who suck at their jobs)
  - Meme Theory: ideology parasitizes the mind, a successful ideology is not configured to be true, but to be easily transmitted and easily believed
  - Gambler's fallacy: we are not owed luck for being unlucky, probability has no memory
  - Do something principle: action creates traction, make your task less intimidating by dividing it into steps and focus only on the next step
  - Lindy Effect: the older something is, the longer it’s likely to be around in the future (good books, intellectual productions, etc.)
  - The Liar's Dividend: politicians can profit from an environment that is saturated with misinformation, as it creates uncertainty among the public
  - Shibboleth: absurd ideological belief is tribal signalling, signifies one considers ideology > truth, reason, or sanity
  - The Potato Paradox: Alice has 100kg of potatoes, which are 99% water. She lets them dry till they are 98% water. What is their new weight? 50kg.
  - Throat-clearing: Before criticizing their own tribe, people feel the need to reaffirm their loyalty to the tribe. "I support X but..."
  - Law of Triviality: Projects that require the least attention tend to get the most.
  - Chilling Effect: When punishment for what people say becomes widespread, people stop saying what they really think and instead say whatever is needed to thrive in the social environment
  - Reiteration Effect: Repeat a lie often enough and it becomes the truth
  - Naïve Realism: tendency to believe our perception of the world reflects it exactly as it is, unbiased and unfiltered
  - Purity Spiral: constnat one-upmanship towards moral superiority causes political tribes to gradually become more extreme (Maoist China)
  - Kayfabrication: Politics is pro-wrestling in suits. Opposing parties are collaborators in a greater system, whose choreographed conflict entertains and distracts us from what is really going on.
  - Postjournalism: The new role of the press is not to inform its readers but to confirm what they already believe (tribalism)
  - Curiosity Zone: curiosity occurs not when you know nothing about something, but when you know a bit about it. So learn a little about as much as you can
  - Sorites Paradox: difficulty of defining clear boundaries or thresholds for certain concepts or ideas (paradox of the heap)
  - Brandolini's Law: It takes a lot more energy to refute bullshit than to produce it. (so the world is full of bullshit)
  - Algorithmic Blindspots: We find growth while searching for other things. Algorithms give us exactly what we want on demand, so we never need to search, and never find what we never knew we needed.
  - Longtermism: the view that positively influencing the long-term future is the key moral priority of our time
  - Two minute rule: If a task would take less than two minutes, do it immediately
  - Promethean Gap: Technology is outpacing wisdom; we're changing the world faster than we can adapt to it. We're amassing the power of gods, yet we remain apes
  - Information-Action Ratio: The mark of useful info is that it makes us act differently. Stop mindless scrolling and seek out info that changes you.
  - Gurwinder's Third Paradox: In order for you to beat someone in a debate, your opponent needs to realize they've lost. Therefore, it's easier to win an argument against a genius than an idiot.
  - Media Naturalness Theory: Writing has existed for <2% of human history, so our brains are not evolved for reading; we need vocal/facial cues for context
  - Tilting At Windmills: online people who attack you are just attacking their own imagination, no need to take it personally
  - Principle Of Humanity: Every single person is exactly what you would be if you were them, seek to understand the circumstances that led others to their conclusions
  - Empty Name: We can be convinced that a concept is real by the mere fact that it has a name, but the world is full of names for things that aren't real

Links 🔗

- [40 Useful Concepts You Should Know - by Gurwinder](https://gurwinder.substack.com/p/40-useful-concepts-you-should-know)

## Day 105: Dec 13, 2022

- Analytics Engineering

  - act as the bridge between data engineers and data analysts/business users. Some of their jobs are:
    - To build well tested, up to date and documented datasets that the rest of the company can use to answer their own questions.
    - Apply software engineering best practices in their data models like version control and CI/CD.
    - Act as facilitators between the business and technical teams and translate strategic business needs into data models and data visualizations.
  - Skills required

    - SQL: strong SQL fluency in aggregation, joins, case statements, CTE and window functions
    - Python: decent knowledge of Python in data types, data structures, if and for loop and how to create functions
    - Data modelling: the process of structuring your raw data into analytics ready format (Star schema, one big Table, data vault)
    - Cloud data warehouse: Google BigQuery, Amazon Redshift and Snowflake
    - Version control: track different versions of your codes and collaborate with other developers
    - data transformation: structuring and reformatting your raw data into data models, involves integrating transactional data (sales, cost, etc) with operational data (name, place, etc) (popular: dbt)
    - data quality testing: set up a proper data quality test and ensure that the data has been tested thoroughly before presenting them to the business users
    - data documentation and lineage:
      - doc: information about your data that ranges from raw schema information to user-supplied information ([dbt data doc](https://docs.getdbt.com/docs/collaborate/documentation))
      - lineage: helps technical and business users to understand how the data flows from data sources to consumption by visualizing them with directed acyclic graphs (DAGs) ([dbt Data lineage](https://docs.getdbt.com/terms/data-lineage))
    - data orchestration: process of gathering the data together from disparate sources and preparing them for data analysis, use tools to automate, schedule and monitor the creation of your dbt models in staging and production environment.
    - CI/CD: used to deploy dbt models to production (If you don't already have Airflow running in production)
    - Data visualization: responsible for developing dashboards with BI tools such as Tableau, Looker or PowerBI
    - Communications: Becoming a better communicator is a skill, not a talent

Links 🔗

- [Becoming An Analytics Engineer in 2023: A Data Analyst Guide](https://medium.com/@baluramachandra90/becoming-an-analytics-engineer-in-2023-a-data-analyst-guide-1faf6d1cc89c)
- [What is Analytics Engineering?](https://www.getdbt.com/what-is-analytics-engineering/)
- [SQL Style Guide | GitLab](https://about.gitlab.com/handbook/business-technology/data-team/platform/sql-style-guide/)
- [Python Guide | GitLab](https://about.gitlab.com/handbook/business-technology/data-team/platform/python-guide/)
- [The Data Warehouse Toolkit: The Definitive Guide to Dimensional Modeling, 3rd Edition [Book]](https://www.oreilly.com/library/view/the-data-warehouse/9781118530801/)
- [Version control basics | dbt Developer Hub](https://docs.getdbt.com/docs/collaborate/git/version-control-basics)
- [Learn Analytics Engineering with dbt | dbt Learn](https://courses.getdbt.com/collections)
- [Template Designer Documentation (Macros and jinja)](https://jinja.palletsprojects.com/en/3.1.x/templates/#list-of-control-structures)
- [Data Testing: What and When to Test](https://www.getdbt.com/analytics-engineering/transformation/data-testing/#why-is-testing-a-necessary-part-of-an-analytics-workflow)
- [Learn Airflow with astronomer](https://docs.astronomer.io/learn)

## Day 106: Dec 14, 2022

- Carbon Dioxide Tolerance Test
  - (1) Take three easy, calm breaths in and out through the nose
  - (2) Take one full nasal inhale, filling the abdomen, upper chest, and lungs
  - (3) Start a timer as you exhale through the nose, as slow as possible
  - (4) Stop the timer when you run out of air, swallow, or feel that you must take a breath
- Interpretation
  - \>80 seconds: Elite (Advanced pulmonary adaptation, excellent breathing control, excellent stress control)
  - 60-80 seconds: Advanced (Healthy pulmonary system, good breathing control, relatively good stress control)
  - 40-60 seconds: Intermediate (Generally improves quickly with focus on CO2 tolerance training)
  - 20-40 seconds: Average (Moderate to high stress/anxiety state, breathing mechanics need improvement)
  - <20 seconds: Poor (Very high anxiety and stress sensitivity, mechanical restriction possible, poor pulmonary capacity)

Links 🔗

- [The CO2 Tolerance Test and Why You Should Be Working on Your Lungs - Crux Fitness](https://www.cruxfit.com/the-co2-tolerance-test-and-why-you-should-be-working-on-your-lungs/)
- [Carbon Dioxide Tolerance Test (CO2TT) - Breathe Your Truth](https://breatheyourtruth.com/carbon-dioxide-tolerance-test-co2tt/)

## Day 105: Dec 15, 2022

- Read intuitive python development on the plane
  - learned about debugging with pdb
  - protobuff and JSON > pickle
  - threads and processes
  - default arguments
  - collections module
  - installing pip securely with hash

Links 🔗

- [Intuitive Python: Productive Development for Projects that Last by David Muller](https://pragprog.com/titles/dmpython/intuitive-python/)

## Day 106: Dec 16, 2022

- process vs thread
  - what's a program?
    - an executable file containing code or set of process instruction that is stored in disk
    - when the code in a program is loaded into memory and executed by processor, it becomes a process
  - active process
    - includes resources - CPU, RAM, I/O managed by OS
    - ex: processor registers, program counters, stack pointers, memory pages (technically under threads)
    - important property: each process has it's own memory space, one process cannot corrupt another process, meaning when one process malfunctions, other process can still run.
    - ex: google chrome tabs
  - threads
    - unit of execution within a process, and it can have many threads
    - each thread has it's own stack, and is possible to communicate between each other using a shared memory address space, which means one misbehaving thread brings down the whole process
  - context switching
    - how does OS run a thread/process?
    - Os stores states of current running process, and resumes state, repeat. (expensive process!)
    - faster to switch context between threads than context, because no need to switch out virtual memory pages, one of the most exp operation
    - how to reduce it?
      - fibers and coroutines
      - trade complexity for even lower context switching
      - coperatively scheduled, where you yield control for oters to run, so app itself handles task scheduling, responsibility of application to make sure a long-running task is broken up.

Links 🔗

- [Process vs Thread | Popular Interview Question - YouTube](https://www.youtube.com/watch?v=4rLW7zg21gI)
- [Thread vs Process in Python](https://superfastpython.com/thread-vs-process/)

## Day 107: Dec 17, 2022

- how to stop overthinking
  - overthinking = genetic predisposition + stressful precipitating events
  - (1) awareness
    - occurs when you shift your attention to your inner and outer experience without judgement, clicning, or resisting
    - write a stress journal and take note of your thoughts and feelings, write down your stress level systematically so you can analyze, evaluate, and know how to maange them
  - (2) time and energy management
    - learn to not prioritize stressful activites but rest and relaxation
    - relax by taking time out to have fun or just spend time doing nothing
    - set SMART goals (specific, measurable, attainable, relevat, time-bound)
    - set time limit for completed goals and use time blocking to avoid multitasking
  - (3) reduce anxiety
    - autogenic training: tell yourself "I am alert" while breathing slowly and steadily. feel sensation of what you're saying in various parts of your body
    - guided imagery and visualization: imagine yourself happy and enjoying it and create a story for 5-20 minutes.
    - progressive muscle relaxation: relax muscles by first tensing them up from head to toe or vice versa
  - (4) stop negative thinking
    - mind, body, and emotion are tightly connected
    - identify negative thought patterns and replace them with positive attitudes
    - common cognitive distortion : black and white thinking, perceiving everything in extremes as either terrible or heavenly.

Links 🔗

- [Stop Overthinking: 23 Techniques to Relieve Stress, Stop Negative Spirals, Declutter Your Mind, and Focus on the Present by Nick Trenton](https://www.goodreads.com/en/book/show/57280624-stop-overthinking)

## Day 108: Dec 18, 2022

- techniques to remember everything
  - (1) memory palace:
    - ad Herennium: two types of memory: natural and artificial. Natural is your innate hardware, artificial is the software you run on your hardware
    - artificial memory has two components
      - image (actual contents one wishes to remember)
      - places (where images are stored)
    - how: form a space in the eye of your mind, a place you can picture easily and fill that place with images representing what you want to remember
    - rememeber the image of each item multi-sensorily, imagine how it feels to touch and smell it, and if it's edible, how it tastes. the more vivid the image the stronger
  - (2) break into chunks
    - George Miller: we can only think about 7 things at a time
    - when a new thing enters the head, it exists in temporary limbo (working memory)
      - working memory: collection of brain systems that hold on to things happening in our consciousness at the present moment
    - to commit info -> long term memory, it has to be repeated over and over again.
    - Chunking: decreasing the number you have to remember by increasing the size of each item.
      - ex: remembering 120741091101 vs 120, 741, 091, 101 vs 12/07/41 and 09/11/01
    - (3) use vivid images to remember things
      - associate a person's name with something you can clearly imagine
      - "all our memories are linked together in a web of associations" - Joshua Foer
      - Baker/baker paradox: the vivid imagery of a baker helped people rmbr Baker surname
      - tip: turn Bakers -> bakers, Foers -> fours, Raegans -> ray guns
    - (4) depict long text with images or emotions
      - methodical : using images to represent each word
      - emotional : breaking down a long text into small chunks and assign a serios of emotions
    - (5) major system
      - code that converts numbers into consonant phonetic sounds -> turned to vivid imagery
      - PAO system (person-action-object), used by US memory championships
        - every two number from 00 to 99 is a single image of a person performing an action on an object

Links 🔗

- [Moonwalking with Einstein: The Art and Science of Remembering Everything by Joshua Foer](https://www.goodreads.com/book/show/6346975-moonwalking-with-einstein)

## Day 109: Dec 19, 2022

- McKinsey Hypothesis-Driven Framework
  - (1) gather facts
    - take a few hours to understand the problem
    - ask:
      - what are the causes?
      - How often does it happen?
      - What solutions have you tried?
      - What solutions have failed?
      - who else should I talk to?
  - (2) generate initial hypothesis
    - problems remain unsolved when people look for solution without direction
    - ask: "Based on initial facts, what do I believe is causing this?"
  - (3) build issue tree
    - Solving one big problem is hard, but solving 20 small problems is easy
    - break down to issue 1 (1.1, 1.2, 1.3), 2, 3
  - (4) understand big picture
    - zoom-out again, take a look at the big picture
    - "What's causing the main problem?"
    - How can this be fixed?
    - summarize into one clear recommendation
  - (5) set the stage
    - People don’t know how you came up with your recommendation.
    - SPQA framework
      - Situation : what's the problem?
      - Problem : why is this a problem?
      - Question : client's main question
      - Answer : what do you recommend
  - (6) convince the stakeholders
    - use pyramid principle
      - restate reccomendation
      - show supporting arguments
      - finish with evidence
  - (7) make impact clcear
    - explain the problem will go away after implementing the solution
      - Whats the step-by-step solution plan?
      - What will the outcome look like?
      - What steps can be taken for extra impact?

Links 🔗

- [Julia MacDonald on Twitter: "McKinsey is paid to solve complex problems. They created a process that helps find solutions: The Hypothesis-Driven Framework. Let’s dive in.👇" / Twitter](https://twitter.com/julia_m_mac/status/1603727231116890118?s=20&t=8FP0B9c6TZGMQ5WM4K0urw)

## Day 120: Dec 20, 2022

- Pareto Principle
  - a minority of causes, inputs, or efforts are responsible for the majority of the results
  - maximizing profits
    - Business: focus serving the top 20% and make efforts for reducing losses from bottom 80%. Study your competitors.
    - Sales: target marketing on 20%, set standards for sales team from top performers, lay off underperforming workers.
  - productivity
    - 80% of our results come from 20% of time put in
    - there are specific times each of us are highly productive, figure out yours from previous productive days and figure out the following:
      - did you work day or night?
      - what happened day before? how was your sleep?
      - how did you handle distractions
    - document your thoughts and use recurring points as pointers to productivity
    - improve happiness level
      - two paper: happiness and achieve island
      - identify things that made you happy and things you achieved
  - Relationships
    - they influence who we become, choose them wisely
    - studies show we have limited space for close relationship, few people account for 20% of relationships, but make up for 80% of the value.
    - 5 attributes determining if relationship will work or not
      - mutual enjoyment of each other's company
      - respect
      - shared experience
      - reciprocity
      - trust

Links 🔗

- [The 80/20 Principle: The Secret to Achieving More with Less by Richard Koch](https://www.goodreads.com/book/show/181206.The_80_20_Principle)

## Day 121: Dec 21, 2022

- Make your bed book CTAs
  - (1) Start each day with a task completed. It will help you feel more capable
  - (2) Make as many friends as possible and never forget your success depends on others
  - (3) Respect everyone who crosses your path. It will teach you dignity
  - (4) Make peace with the imperfection of life, learn how to react to and deal with the problems of life
  - (5) Next time you fail hard, don't make things worse by humiliating yourself. Inhale and shift the focus to things that have potential to make your life better.
    - "life is a paticular pattern of success and failure, wihout the latter you won't know what the former feels like."
  - (6) Dare to take (calculated) risks, they help you extend your experience and mature.
  - (7) Make a top 5 list of things that seemed to break you.
    - Write down what helped you to overcome them, what you learned about yourself from them, and how it made you stronger. For it did.

Links 🔗

- [Make Your Bed: Little Things That Can Change Your Life...And Maybe the World by William H. McRaven](https://www.goodreads.com/book/show/31423133-make-your-bed)

## Day 122: Dec 22, 2022

- Best way to apologize
  - what are apologies for?
    - it's not for you, it's not about getting forgiveness and moving on
    - it's about expressing remorse and accepting accountability
  - (1) taking victim's perspective
    - ask other party how you made them feel to recognize your wrong doing
  - (2) Accepting responsibility
    - accept how your actions caused harm, frame your apology around it
  - (3) making concrete offer of repair
    - indicate exactly how you'll change and repair the damage caused by your expense.

Links 🔗

- [The best way to apologize (according to science) - YouTube](https://www.youtube.com/watch?v=q-ApAdEOm5s)

## Day 123: Dec 23, 2022

- On Writing well
  - (1) strip every sentence to it's cleanest components
    - Simplify, simplify.
    - Clear thinking becomes clear writing; one can’t exist without the other.
    - The writer must constantly ask himself: What am I trying to say? Have I said it? Is it clear to someone encountering the subject for the first time?
  - (2) be yourself
    - write in the first person: to use "I" and “me" and “we" and “us."
    - relax, and have confidence.
  - (3) write for yourself
    - Never say anything in writing that you wouldn’t comfortably say in conversation.
  - (4) Leave readers with one provocative thought
    - "every successful piece of nonfiction should leave the reader with one provocative thought that he or she didn’t have before"
  - (5) Create a good hook
    - The most important sentence in any article is the first one. If it doesn’t induce the reader to proceed to the second sentence, your article is dead. And if the second sentence doesn’t induce him to continue to the third sentence, it’s equally dead. Of such a progression of sentences, each tugging the reader forward until he is hooked, a writer constructs that fateful unit, the “lead."
    - how?
      - cajole him with freshness, or novelty, or paradox, or humor, or surprise, or with an unusual idea, or an interesting fact, or a question.
      - provide hard details that tell the reader why the piece was written and why he ought to read it
      - don’t dwell on the reason. Coax the reader a little more; keep him inquisitive
  - (7) always collect more material than you will use
    - look for your material everywhere, not just by reading the obvious sources and interviewing the obvious people.
  - (6) Create a good ending
    - The perfect ending should take your readers slightly by surprise and yet seem exactly right.
    - simple rule: when you’re ready to stop, stop. If you have presented all the facts and made the point you want to make, look for the nearest exit
    - what usually works best is a quotation. Go back through your notes to find some remark that has a sense of finality, or that’s funny, or that adds an unexpected closing detail.
    - Surprise is the most refreshing element in nonfiction writing
  - bits & pieces
    - Use active verbs unless there is no comfortable way to get around using a passive verb.
    - Short is better than long.
    - Most adverbs are unnecessary.
    - Most adjectives are also unnecessary.
    - Prune out the small words that qualify how you feel and how you think and what you saw: "a bit", "a little", "sort of", "kind of", "rather", "quite", "very", "too", "pretty much", "in a sense" and dozens more. They dilute your style and your persuasiveness.
    - Always use "that" unless it makes your meaning ambiguous.
    - Rewriting is the essence of writing well: it’s where the game is won or lost.
    - No subject is too specialized or too quirky if you make an honest connection with it when you write about it.
  - Write as well as you can
    - If you would like to write better than everybody else, you have to want to write better than everybody else
    - You must take an obsessive pride in the smallest details of your craft
    - purposes that writers serve must be their own
    - What you write is yours and nobody else’s
    - Writing well means believing in your writing and believing in yourself, taking risks, daring to be different, pushing yourself to excel. You will write only as well as you make yourself write.

Links 🔗

- [On Writing: A Memoir of the Craft by Stephen King](https://www.goodreads.com/book/show/10569.On_Writing)
- [Summary & Notes](https://www.grahammann.net/book-notes/on-writing-well-william-zinsser)

## Day 124: Dec 24, 2022

- The Bento methodology
  - an ineffective to-do list = too many task in limited time + filling it with low-value tasks
  - bento methodology tackles this by having
    - A handful of well-intended things to do
    - A mix between high and low-value tasks in the list
  - it can hold up to 3 tasks
    - 1 large task: Task that requires deep focus and works towards your goals. (>90 mins)
    - 1 medium task: Task that is usually busy work like organising reports. (<45 mins)
    - 1 short task: Daily chores you must do at home or work, like paying your rent. (~15 mins)
  - Account for energy levels
    - Our body energy levels fluctuate over the day in the form of Peak-Trough-Rebound
    - peak = mind and body are at their highest functioning form
    - through = exhausted from work during your peak period and need time and rest to replenish.
    - rebound = body is recovering from the low-energy trough and is ready to get back in focus mode for another few hours.
    - don't waste high-energy time on non-challenging tasks like answering emails or collecting documents for a visa application during the rebound period.
  - 5-tasks to-do list system
    - 2 high-value or deep work tasks assigned for the peak and rebound
    - 2 medium-value or admin work tasks assigned for the trough
    - 1 low-value or busy work task aimed for the trough or towards the end of the day
    - in Todoist:
      - P1: Deep or high-value work. Example: Writing an article or a report, solving complex problems at work, learning a new concept.
      - P2: Admin work. Example: Collating data for analysis, organising invoices to pay, sending reports to a coworker.
      - P3: Business or household chores. Example: Catching up on emails, taking your dog for a walk, renewing your car insurance.
  - preliminary assessment
    - observe your energy levels, mentally note when you feel most energetic during the day.
    - Understanding when your energy levels peak and dip will help you assign the right tasks at the right moment each day.

Links 🔗

- [Crafting To-Do Lists That Spark Productivity and Joy](https://hulry.com/productive-to-do-list/)
- [How I Finally Made Sense of Todoist’s Priority Levels](https://hulry.com/todoist-priority-levels-moscow/)

## Day 125: Dec 25, 2022

- what's next for AI
  - (1) multipurpose chatbots
    - OpenAI is interested in combining different modalities—such as image or video recognition—with text
    - Imagine being able to ask a chatbot what’s in an image, or asking it to generate an image, and have these interactions be part of a conversation so that you can refine the results more naturally than is possible with DALL-E.
    - glimpse of this
      - DeepMind’s Flamingo, a “visual language model” revealed in April, which can answer queries about images using natural language
      - Gato, a “generalist” model that was trained using the same techniques behind large language models to perform different types of tasks, from describing images to playing video games to controlling a robot arm.
    - GPT-4 = power of the best language and image-making AI (and more) in one package
    - downsides
      - inability to tell fact from fiction
      - penchant for prejudice
  - (2) AI Act
    - EU's sweeping AI law: include bans on AI practices deemed detrimental to human rights, such as systems that score and rank people for trustworthiness
    - facial recognition
      - The use of facial recognition in public places will also be restricted for law enforcement in Europe
      - momentum to forbid that altogether for both law enforcement and private companies, although a total ban will face stiff resistance from countries that want to use these technologies to fight crime
    - user data and privacy
      - Federal Trade Commission is also closely watching how companies collect data and use AI algorithms
      - a new law to hold AI companies accountable when their products cause harm, such as privacy infringements or unfair decisions made by algorithms.
    - deepfake
      - In China, authorities have recently banned creating deepfakes without the consent of the subject
      - Europeans want to add warning signs to indicate that people are interacting with deepfakes or AI-generated images, audio, or video
    - All these regulations could shape how technology companies build, use and sell AI technologies. However, regulators have to strike a tricky balance between protecting consumers and not hindering innovation
    - if new laws are implemented correctly, 2023 could usher in a long-overdue era of AI development with more respect for privacy and fairness.
  - (3) open source & startups > big tech for AI research
    - first community-built, multilingual large language model, BLOOM, released by Hugging Face
    - explosion of innovation around the open-source text-to-image AI model Stable Diffusion, which rivaled OpenAI's DALL-E 2.
    - AI research is expensive, and as purse strings are tightened, companies will have to be very careful about picking which projects they invest in—and are likely to choose whichever have the potential to make them the most money, rather than the most innovative, interesting, or experimental ones
    - Startups and academia could become the centers of gravity for fundamental research
    - flashy new upstarts working on generative AI are seeing a surge in interest from venture capital funds.
  - (4) Big Pharma + AI = 🚀
    - DeepMind's AlphaFold
      - an AI that can predict the structures of proteins (the key to their functions), has cleared a path for new kinds of research in molecular biology, helping researchers understand how diseases work and how to create new drugs to treat them.
    - Meta's ESMFold
      - a much faster model for predicting protein structure—a kind of autocomplete for proteins, which uses a technique based on large language models
    - Both produced structures for hundreds of millions of proteins, including all that are known to science, and shared them in vast public databases
    - "makes looking up new protein structures almost as easy as searching the web"
    - DeepMind has spun off its biotech work into a separate company, Isomorphic Labs
    - hundreds of startups exploring ways to use AI to speed up drug discovery and even design previously unknown kinds of drugs
    - 19 drugs developed by AI drug companies in clinical trials (up from zero in 2020), with more to be submitted in the coming months

Links 🔗

- [What's next for AI | MIT Technology Review](https://www.technologyreview.com/2022/12/23/1065852/whats-next-for-ai/)

## Day 126: Dec 26, 2022

- watched 30% of fast.ai lecture 1
- learned about Vint Cerf and his contribution towards the internet

Links 🔗

- [Vint Cerf Helped Create the Internet on the Back of an Envelope - WSJ](https://archive.ph/k1j4H)

## Day 127: Dec 27, 2022

- functional programming week 1
  - substitution model
    - all evaluation does is reduce an expression -> value
    - can be applied to all expression as long as they have no side effects (purely functional)
    - formalized in λ-calculus
  - call-by-name: functions applied to unreduce arguments (ignores unused parameters)
  - call-by-value: functions applied to reduced arguments (evaluates every function arg only once)
    - ex: `def test(x: Int, y: Int) = x \* x
      - test(2, 3) - same
      - test(3+4, 8) - CBV
      - test(7, 2\*4) - CBN
      - test(3+4, 2\*4) - same
  - Scala is call-by-value by default
  - to use call-by-name, add => like `(y: => Double)`

Links 🔗

- [Functional Programming Principles in Scala notes](https://ivy-zhou.github.io/notes/2020/09/02/progfun.html)

## Day 128: Dec 28, 2022

- Conformal Prediction
  - a method that takes an uncertainty score and turns it into a rigorous score
    - heiristic uncertainty -> conformal prediction -> rigorous uncertainty
  - "rigorous" = output has probablistic guarantees of covering the true outcome
  - it turns point predictions into prediction sets. for multi-class classification, it turns class output to a set of classes
    - ex: {cat} (most likely) -> {cat, lion, hamster} (prediction set with probabilistic coverage guarantees)
  - many advantages
    - distribution-free : only assumption is data points are exchangable
    - model-agnostic: can be applied to any predictive model
    - coverage guarantee : resulting prediction set comes with guarantees of covering true outcome with certain probability.
  - start with the guarantee
    - produce a set of predictions for every bean that covers the true class with 95% probability? Seems to be a matter of finding the right threshold
    - use the model “probabilities” to construct a measure of uncertainty: `s_i = 1 - f(x_i)[y_i]`
      - A bit sloppy notation for saying that we take 1 minus the model score for the true class (if the ground truth for bean number 8 is “Seker” and the probability score for Seker is 0.9, then s_8 = 0.1).
    - process:
      - start with fresh data
      - compute the scores s_i
      - sort the scores from low (certain) to high (uncertain)
      - compute the threshold q where 95% of the s_i’s are lower (=95% quantile)
    - How is that threshold useful? We know that for bean uncertainties s_i below 0.999 (equivalent to class “probabilities” > 0.001) we can be sure that with a probability of 95% we have the correct class included.
    - An important part of conformal prediction is that the data (X_calib, y_calib) used for finding that threshold is not used for training the model.
      - For new data points the data scientist can turn the prediction into prediction sets by using the threshold: `prediction_sets = (1 - model.predict_proba(X_new) <= qhat)`
  - Coverage
    - percentage of prediction sets that contain the true label
    - While conformal predictors always guarantee marginal coverage, conditional coverage is not guaranteed. Adaptive conformal predictors approximate conditional coverage.
    - Add up all the probabilities, starting with the highest one, until we reach the true class. This algorithm is called “Adaptive Prediction Sets”.
  - Recipe for CP
    - Training
      - Split data into training and calibration
      - Train model on training data
    - Calibration
      - Compute uncertainty scores (aka non-conformity scores) for calibration data
      - Sort the scores from certain to uncertain
      - Decide on a confidence level α (α=0.1 means 90% coverage)
      - Find the quantile q where 1-α (multiplied with a finite sample correction) of non-conformity scores are smaller
    - Prediction (how you use the calibrated score)
      - Compute the non-conformity scores for the new data
      - Pick all y’s that produce a score below q
      - These y’s form your prediction set or interval
  - A big differentiator between conformal prediction algorithms is the choice of the score. In addition, they can differ in the details of the recipe and slightly deviate from it as well.
  - cross-conformal prediction: CP with resampling
    - find a balance between training the model with as much data as possible, but also having access to “fresh” data for calibration.
    - 3 methods
      - Single split: Computation-wise the cheapest. Results in a higher variance of the prediction sets and non-optimal use of data. Ignores variance from model refits. Preferable if refitting the model is expensive.
      - jackknife: Most expensive, since you have to train n models. The jackknife approach potentially produces smaller prediction sets/intervals as models are usually more stable when trained with more data points. Preferable if model refit is fast and/or the dataset is small.
      - CV and other resampling methods: balance between single split and jackknife.

Links 🔗

- [valeman/awesome-conformal-prediction: A professionally curated list of awesome Conformal Prediction videos, tutorials, books, papers, PhD and MSc theses, articles and open-source libraries.](https://github.com/valeman/awesome-conformal-prediction)
- [awslabs/fortuna: A Library for Uncertainty Quantification.](https://github.com/awslabs/fortuna)
- [Week #1: Getting Started With Conformal Prediction For Classification](https://mindfulmodeler.substack.com/p/week-1-getting-started-with-conformal)
- [Week #2: Intuition Behind Conformal Prediction](https://mindfulmodeler.substack.com/p/week-2-intuition-behind-conformal)
- [A Gentle Introduction to Conformal Prediction](https://arxiv.org/pdf/2107.07511.pdf)

## Day 129: Dec 29, 2022

- Models that reason
  - Recent advances in deep learning largely have come by brute force: taking SOTA architecture + scale up compute power, data, and engineering
  - simply scaling up is not going to fill the gap. Instead, building into our models a human-like ability to discover and reason with high-level concepts and relationships between them can make the difference.
  - No. examples to learn learn a new task = sample complexity.
    - It takes a huge amount of gameplay to train a deep learning model to play a new video game, while a human can learn this very quickly. Related issues fall under the rubric of reasoning.
    - A computer needs to consider numerous possibilities to plan an efficient route from here to there, while a human doesn’t
  - Current systems can't reason
    - Humans can select the right pieces of knowledge and paste them together to form a relevant explanation, answer, or plan
    - given a set of variables, humans are pretty good at deciding which is a cause of which
    - current AI is highly confident that their decision is right, even when it’s wrong, can be life-threatening in a self-driving car or medical diagnosis system.
  - generative flow networks (GFlowNets)
    - inspired by the way humans reason through a sequence of steps, adding a new piece of relevant information at each step
    - like reinforcement learning, because the model sequentially learns a policy to solve a problem.
    - like generative modeling, because it can sample solutions in a way that corresponds to making a probabilistic inference
    - ex: think of an interpretation of an image
      - your thought can be converted to a sentence, but it’s not the sentence itself. It contains semantic and relational information about the concepts in that sentence.
      - Generally, we represent such semantic content as a graph, in which each node is a concept or variable.
      - GFlowNets generate such graphs one node or edge at a time, choosing which concept should be added and connected to which others in what kind of relation

Links 🔗

- [GFNOrg/gflownet: Generative Flow Networks](https://github.com/GFNOrg/gflownet)
- [The What, Why and How of Generative Flow Networks | by Matt Biggs | Towards Data Science](https://towardsdatascience.com/the-what-why-and-how-of-generative-flow-networks-4fb3cd309af0)
- [Generative Flow Networks - Yoshua Bengio](https://yoshuabengio.org/2022/03/05/generative-flow-networks/)

## Day 130: Dec 30, 2022

- A personal data timeline
  - we create a lot of data
    - Photos capture our experiences
    - phones record our workouts and locations
    - Internet services log the content we consume and our purchases.
    - we record our want-to lists: desired travel and dining destinations, books and movies we plan to enjoy, and social activities we want to pursue.
    - Soon smart glasses will record our experiences in even more detail.
  - this data is siloed in dozens of applications
    - we often struggle to retrieve important facts from our past and build upon them to create satisfying experiences on a daily basis
  - use our data -> improve health, vitality, and productivity?
    - what if all information we create online were fused in a personal timeline designed to help us stay on track toward our goals, hopes, and dreams?
    - Vannevar Bush envisioned it in 1945, calling it a memex
    - Gordon Bell and his colleagues at Microsoft Research built MyLifeBits, a prototype of this vision
  - Privacy concern
    - Privacy means that your data is available only to you, but if you want to share parts of it
    - No single company has all our data or the trust to store all our data
    - building technology that enables personal timelines should be a community effort that includes protocols for the exchange of data, encrypted storage, and secure processing
  - 2 AI Challenges
    - (1) answering questions over personal timelines
      - question answering requires that we reason explicitly about sets of answers and aggregates computed over them
      - This is the bread and butter of database systems
      - Ex: “what cafes did I visit in Tokyo?” or “how many times did I run a half marathon in under two hours?” requires that we retrieve sets as intermediate answers, which is not currently done in natural language processing
      - Borrowing more inspiration from databases, we also need to be able to explain the provenance of our answers and decide when they are complete and correct
    - (2) our timelines -> improved personal well-being
      - Taking inspiration from the field of positive psychology, we can all flourish by creating positive experiences for ourselves and adopting better habits
      - An AI agent that has access to our previous experiences and goals can give us timely reminders and suggestions of things to do or avoid.
    - "an AI with a holistic view of our day-to-day activities, better memory, and superior planning capabilities would benefit everyone"

Links 🔗

- [MyLifeBits - Microsoft Research](https://www.microsoft.com/en-us/research/project/mylifebits/)
- [Memex - Wikipedia](https://en.wikipedia.org/wiki/Memex)

## Day 131: Dec 31, 2022

- Active Learning
  - what? - enables machine learning systems to generate their own training examples and request them to be labeled
  - can enable machine learning systems to:
    - Adapt to changing conditions
    - Learn from fewer labels
    - Keep humans in the loop for the most valuable/difficult examples
    - Achieve higher performance
  - generative AI -> boom in Active learning
    - with recent advances in generative AI for images and text, when a learning algorithm is unsure of the correct label for some part of its encoding space, it can actively generate data from that section to get input from a human
  - revolutionary approach to ML
    - allows systems to continuously improve and adapt over time
    - Rather than relying on a fixed set of labeled data, it seeks out new information and examples that will help it better understand the problem it is trying to solve
    - more accurate and effective machine learning models, and it could
    - reduce the need for large amounts of labeled data

Links 🔗

- [Active Learning, part 1: the Theory](https://blog.scaleway.com/active-learning-some-datapoints-are-more-equal-than-others/)
- [google/active-learning](https://github.com/google/active-learning)
- [baifanxxx/awesome-active-learning: A curated list of awesome Active Learning](https://github.com/baifanxxx/awesome-active-learning)
- [rmunro/pytorch_active_learning: PyTorch Library for Active Learning to accompany Human-in-the-Loop Machine Learning book](https://github.com/rmunro/pytorch_active_learning)
- [[1702.07956] Generative Adversarial Active Learning](https://arxiv.org/abs/1702.07956)

## Day 132: Jan 1, 2023

- 22 Problems Solved in 2022
  - (1) NASA asteroid program
  - (2) US joined the Kigali Amendment as part of the federal government's effort to phase out problematic carbon substances
  - (3) 50 (extinct) species are returning in record numbers
  - (4) malaria vaccine that can save millions of lives
  - (5) Lyme disease vaccine nearing market return
  - (6) US soccer teams strike monumental deal for gender discrimination
  - (7) free lunches program expand
  - (8) Europe stnadardizing charging ports in 2024
  - (9) US EV tipping point hit (5% of total vehicle sales)
  - (10) $4.7B to plug orphan wells
  - (11) Canada pilots prescribing outdoor time
  - (12) US military suicide sees decline
  - (13) HIV vaccines progressing through trials
  - (14) art museums solve funding issues
  - (15) Battery swap technology 500k users (Gogoro)
  - (16) Ethereum -> Proof of stake model (cuts energy consumption by 99.95%)
  - (17) MLB solves authentication
  - (18) Klamath river set for return (salmons)
  - (19) deepfake detector by Intel (FakeCatcher) 96% accuracy
  - (20) solution for removing PFAs (forever chemical) found
  - (21) US states ban prison slavery from states constitution
  - (22) nuclear fusion breakthrough (more energy produced thatn used)

Links 🔗

- [22 Problems Solved in 2022 - YouTube](https://www.youtube.com/watch?v=c3dDagZMALQ&t=89s)

## Day 133: Jan 2, 2023

- Conversational skills
  - (1) Don’t Interrupt
    - A conversation is best when both parties are interested, engaged, and want to share. If you interrupt you show that you are uninterested and you blunt the other person’s motivation to share
  - (2) Accept, don't seek
    - Accept whatever reaction someone gives you, and treat it as though it were correct. If you said something hilarious and they didn’t laugh, act as if it wasn’t funny. Don't seek for laughs or validation, it puts a huge burden on listeners.
  - (3) gauge interest
    - tell the quickest version of your story, and leave out any tangents or details. This puts the other person in the driver’s seat and lets them ask about the things that most interest them
  - (4) ask questions
    - An ideal conversation is a mix of listening, asking questions, and sharing in a way that allows the other person to politely guide the conversation.
    - You must ask questions so that the other person knows that you are interested in them and what they are saying.
    - Factual questions are good, but questions that deepen the conversation are even better (“Was that as hard as it sounds?” “How did you learn how to do that?” “What made you decide to go that route?”)
  - (5) verbalize when you change your mind
    - People love to teach and persuade others. When someone has taught you something or changed your mind on something, let them know. Say things like: “Wow, I would have never thought to do that, but that’s a great idea.”, “Ok, maybe you’re right. I hadn’t thought about that before.”, “You know, I used to really think X, but you’ve convinced me Y”
  - disagree positively
    - While you want to be an agreeable person, it’s important to share your real thoughts and to express them in proportion with their weight.
    - ex: John tells you he lives in Vegas but you don’t think that you would like living in Vegas. You ask “Vegas? Casinos and desert? What made you choose that?” and not “Wow, I would never live there. It’s too hot and there are no trees.” (this is a miserable conversation)
  - benchmarking
    - "Nice", "Cool", "Ok"
      - If you get a lot of single word answers, you are not keeping the other person interested. Switch topics or talk less
    - 50% talk time
      - Conversations should be just about 50/50, especially if you know each other reasonably well. It is your responsibility to get the ratio correct, either by talking more or by asking more questions to induce the other person to talk more
    - Depth
      - Conversation should frequently be making it to a depth where you are learning more about the other person and they are learning more about you
      - you must be willing to be vulnerable and share things about yourself, and must also have the awareness to ask questions to induce the other person to do the same.

Links 🔗

- [Conversation Skills Essentials – Tynan.com](https://tynan.com/letstalk/)
- [Creating Emotional Connection Handouts.pdf](https://caps.unl.edu/Creating%20Emotional%20Connection%20Handouts.pdf)
- [A low-risk technique for gaining intimacy with people - YouTube](https://www.youtube.com/watch?v=WyKFHd7cSaU&t=108s)

## Day 134: Jan 3, 2023

- top 10 AI papers
  - (1) ConvNeXt
    - purely convolutional arch that outperforms Vision transformers (Swin transformer) and all previous CNN models
    - new default when it comes to using CNN not only for classification, but also object detection and instance segmentation
    - a ResNet-50 base architecture + depthwise convolutions, inverted bottleneck layer designs, AdamW, LayerNorm, and many more
    - modern data augmentation techniques such as Mixup, Cutmix, and others.
  - (2) MaxViT: Multi-axis Vision Transformer
    - early vision transformers suffered from quadratic complexity, many tricks have been implemented to apply vision transformers to larger images with linear scaling complexity.
    - this is achieved by decomposing an attention block into two parts with local-global interaction:
      - local attention ("block attention");
      - global attention ("grid attention").
    - a convolutional transformer hybrid featuring convolutional layers as well.
    - current trend is ViT + CNN -> hybrid architectures
  - (3) Stable Diffusion
    - what are diffusion models?
      - a type of probabilistic model that are designed to learn the distribution of a dataset by gradually denoising a normally distributed variable.
      - This process corresponds to learning the reverse process of a fixed Markov Chain over a length of T.
    - GANS = minimax game between generator and discriminator
    - Diffusion = likelihood-based models trainsed using MLE (help to avoid mode collapse and other training instabilities)
    - paper's novelty is in applying diffusion in latent space using pretrained autoencoders instead of using the full-resolution raw pixel input space of the original images directly.
    - training process
      - phase 1: pretrain autoencoder to encode input images into lower-dimensional latent space to reduce complexity
      - phase 2: train diffusion model son latent representation of pretrained autoencoder
    - another contribution of paper is the cross-attention mechanism for general conditioning
    - capable of not just unconditional generation, but also inpainting, class-conditional image synthesis, super-resolution, and text-to-image synthesis
  - (4) Gato
    - A generalist agent capable of performing over 600 tasks, ranging from playing games to controlling robots.
  - (5) Training Compute-Optimal Large Language Models.
    - To achieve optimal computation during training, it's necessary to scale both the model size and the number of training tokens by the same factor.
    - Chinchilla that outperformed Gopher using 4 times fewer parameters and 4 times more data.
  - (6) PaLM: Scaling Language Modeling with Pathways:
    - shows impressive natural language understanding and generation capabilities on various BIG-bench tasks.
    - To some extent, it can even identify cause-and-effect relationships.
  - (7) Whisper:
    - Robust Speech Recognition via Large-Scale Weak Supervision paper
    - It was trained for 680,000 hours on multilingual tasks and exhibits robust generalization to various benchmarks
  - (8) Revisiting Pretraining Objectives for Tabular Deep Learning.
    - highlights and reminds us how important it is to pretrain models on additional (typically unlabeled) data. (You can't easily do this with tree-based models like XGBoost.)
  - (9) Why do tree-based models still outperform deep learning on tabular data?
    - tree-based models (random forests and XGBoost) outperform deep learning methods for tabular data on medium-sized datasets (10k training examples). But the
    - gap between tree-based models and deep learning becomes narrower as the dataset size increases (here: 10k -> 50k).
  - (10) Evolutionary-scale prediction of atomic level protein structure with a language model.
    - proposed the largest language model for predicting the three-dimensional structure of proteins to date.
    - faster than previous methods while maintaining the same accuracy.
    - created the ESM Metagenomic Atlas, the first large-scale structural characterization of metagenomic proteins, featuring over 617 million structures.

Links 🔗

- [Ahead of AI #4: A Big Year For AI - by Sebastian Raschka](https://magazine.sebastianraschka.com/p/ahead-of-ai-4-a-big-year-for-ai)
- [[2201.03545] A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)
- [[2204.01697] MaxViT: Multi-Axis Vision Transformer](https://arxiv.org/abs/2204.01697)
- [[2112.10752] High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)
- [[2205.06175] A Generalist Agent](https://arxiv.org/abs/2205.06175)
- [[2203.15556] Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)
- [[2204.02311] PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/abs/2204.02311)
- [[2212.04356] Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356)
- [[2207.03208] Revisiting Pretraining Objectives for Tabular Deep Learning](https://arxiv.org/abs/2207.03208)
- [[2207.08815] Why do tree-based models still outperform deep learning on tabular data?](https://arxiv.org/abs/2207.08815)
- [Evolutionary-scale prediction of atomic level protein structure with a language model | bioRxiv](https://www.biorxiv.org/content/10.1101/2022.07.20.500902v3)

## Day 135: Jan 4, 2023

- 5 mindset to be confident
  - (1) attract others with personality, the most attractive trait is confidence
    - make people feel special: be authentic and focus on what people say, think "what do i admire about this person?"
    - say more than necessary: if people ask how are you? say how your day went and something that happened, make them feel comfortable and able to latch on to something
    - emanate warmth: smile with your eyes
    - make use of touch: exercise judgement, only appropriate touch, talk closely, look in their eyes
    - speak with calm, gentle voice
  - (2) live in accordance with your values
    - make a list of values, access how you've been living by them, and start fixing what you haven't been living up to
    - figure out your main values, the ones you truly live for
    - note down what others who have those values do, follow them, evaluate in your own behaviours and actions
  - (3) act like you're not nervous
    - everyone is awkward at some level, don't judge yourself for it
    - don't cut out mistakes, but laugh about them, it shows personality and you're not devoid of humor
    - don't take yourself too seriously, it shows you're insecure because you care what others think
  - (4) put in the work, then aim for the top
    - biggest lie is you shouldn't go for your dreams
    - aim for jobs you think you want, not what you can get
    - don't settle for relationships
  - (5) stop caring what people think about you
    - the only opinion that matters is yourself
    - some people are going to dislike you
    - don't take criticism from people you wouldn't take advice from
    - if you're not ruffling some feathers, then you're doing something wrong
- Attract the right people
  - (1) list all traits that you want around you (for partner, friend, family member)
  - (2) highlight traits that you don't have
  - (3) identify steps you need to take to bridge those gaps (SMART goals: specific, measurable, actionable, relevant and time based)
  - (4) put it into action (weekly goals, check in with yourself)

Links 🔗

- [5 Mindsets to be Confident & Well-Liked - YouTube](https://www.youtube.com/watch?v=4P9Qp2Q3mLo)
- [How to attract the right people by working on yourself - YouTube](https://www.youtube.com/watch?v=Mj9J_zsYvtE)

## Day 136: Jan 5, 2023

- Tensors

  - x - Scalar, 0D Tensor (single value)
  - [ x, y ] - Vector, 1D Tensor, 1D Array or simply Array (multiple values in one row)
  - [ [x, y, z], [a, b, c] ] - Matrix, 2D Tensor, 2D Array (multiple values in multiple rows and columns)
  - they are used as a useful way of packing large volumes of data in a single variable that is represented like nD Tensor, and then perform your regular math on these tensors
  - Since GPUs are optimized to do math on geometrical objects (in a 3D game, for example), they are particularly well suited for the job of making calculations that involve complex data represented as Tensors
  - PyTorch
    - in-memory : `x = torch.empty(3, 4)`
    - initialize values : `torch.zeros()`, `torch.ones()`, `torch.rand()`
    - seed: `torch.manual_seed(69)`
    - shapes: `torch.empty(2, 2, 3)` = 3d tensor of 2x2x3 shape
    - same shape: `torch.*_like()`
    - already have data: `torch.tensor()`
    - data types: `torch.int16`, `float64`
    - math: `torch.abs`, `torch.ceil`, `torch.floor`, `torch.clamp`
    - trig: `torch.sin`, `torch.asin`
    - bitwise: `torch.bitwise_xor`
    - equality: `torch.eq`
    - reductions: `torch.max`, `torch.std`, `torch.prod`, `torch.unique`
    - vector: `torch.cross`, `torch.matmul`, `torch.svd`
    - ater in place: append `_` to function (ex: `torch.add_(b)`)
    - specify tensor to receive output: `out = c`
    - copy tensor: `.clone()`
    - turn on autograd : `requires_grad=True`
    - check GPU: `if torch.cuda.is_available(): my_device=torch.device('cuda')`
    - query count: `torch.cuda.device_count()`
    - device handle: `torch.rand(2,2,device=my_device)`
    - add dimensions of size 1: `x.unsqueeze(0)` (adds a zeroth dimension)
    - remove dimension of size 1: `x.squeeze(1)`
    - reshape: `reshape(features x width x height)`

- Broadcasting Tensors rules
  - Each tensor must have at least one dimension - no empty tensors.
  - Comparing the dimension sizes of the two tensors, going from last to first:
    - Each dimension must be equal, or
    - One of the dimensions must be of size 1, or
    - The dimension does not exist in one of the tensors

Links 🔗

- [Visualization of tensors - part 1 - YouTube](https://www.youtube.com/watch?v=YxXyN2ifK8A)
- [Tensors for Neural Networks, Clearly Explained!!! - YouTube](https://www.youtube.com/watch?v=L35fFDpwIM4)
- [Multi-Dimensional Data (as used in Tensors) - Computerphile - YouTube](https://www.youtube.com/watch?v=DfK83xEtJ_k)
- [Introduction to PyTorch Tensors — PyTorch Tutorials 1.13.1+cu117 documentation](https://pytorch.org/tutorials/beginner/introyt/tensors_deeper_tutorial.html)

## Day 137: Jan 6, 2023

- finished fastai lesson 1 and chapter 1
  - universal approximation theorem:
    - NNs can theoretically represent any mathematical function. However, practically, due to limits of available data and computer hardware, it is impossible, but can get very close.
  - Positive feedback loop:
    - ex: a model that predict crimes. use current arests as training data (biased) -> build model -> use model -> increase arrests in that area -> used in training data -> more bias
  - Segmentation:
    - pixelwise classification problem, we predict a label for every single pixel in the image. this provides a mask for which parts of image corresponds with label
  - metric vs loss:
    - metric measures quality of model's prediction with validation set, loss is measure of performance of model, meant for optimization algorithms (SGD) to efficiently update model parameters. metrics are human-interpretable
  - head of model:
    - in pretrained model, later layers of model are useful for original task, but replaced with one/more new layers with randomized weights approrpiate for the current dataset for fine-tuning, these new layers are head

Links 🔗

- [Fastbook Chapter 1 questionnaire solutions (wiki) - Part 1 (2020) - fast.ai Course Forums](https://forums.fast.ai/t/fastbook-chapter-1-questionnaire-solutions-wiki/65647)
- [1. Your Deep Learning Journey - Deep Learning for Coders with fastai and PyTorch [Book]](https://www.oreilly.com/library/view/deep-learning-for/9781492045519/ch01.html)
- [fastbook/01_intro.ipynb at master · fastai/fastbook](https://github.com/fastai/fastbook/blob/master/01_intro.ipynb)

## Day 138: Jan 7, 2023

- From model to production
  - When selecting a project, the most important consideration is data availability
  - Computer vision
    - if what you are trying to classify are sounds, you might try converting the sounds into images of their acoustic waveforms and then training a model on those images
  - Text (NLP)
    - text generation models will always be technologically a bit ahead of models for recognizing automatically generated text
  - text + images
    - a model can be trained on input images with output captions written in English, and can learn to generate surprisingly appropriate captions automatically for new images
    - an automatic system can be used to identify potential stroke victims directly from CT scans, and send a high-priority alert to have those scans looked at quickly, but require human-in-the-loop
  - tabular data
    - Tree models are still faster, but DL greatly increase the variety of columns that you can include—for example, columns containing natural language (book titles, reviews, etc.) and high-cardinality categorical columns (large number of discrete choices, such as zip code or product ID).
  - recommender systems
    - special type of tabular data, generally have a high-cardinality categorical variable representing users, and another one representing products (or something similar)
    - A company like Amazon represents every purchase that has ever been made by its customers as a giant sparse matrix, with customers as the rows and products as the columns.
    - if customer A buys products 1 and 10, and customer B buys products 1, 2, 4, and 10, the engine will recommend that A buy 2 and 4.
    - nearly all machine learning approaches have the downside that they tell you only which products a particular user might like, rather than what recommendations would be helpful for a user
  - drivetrain approach
    - objective -> levers (controllable inputs) -> data -> models (how levers influence objective)
    - ex: Google
      - objective = "show the most relevant search result."
      - levers = ranking of the search results
      - data = implicit information regarding which pages linked to which other pages
      - models take both the levers and any uncontrollable variables as their inputs; the outputs from the models can be combined to predict the final state for our objective
    - ex: recommendation systems
      - objective = drive additional sales
      - lever = ranking of recs
      - data = randomized experiments to collect data that is representative
      - model = build two, conditional on seeing / not seeing recs
        - difference between the two probability is utility function for a given rec to customer
        - low when: algo rec familiar book customer already rejected (both are low) or book that they would have bought even without recs (both ar elarge, cancel each other out)
  - practical implementation of model is more than just training the model. You’ll often need to run experiments to collect more data, and consider how to incorporate your models into the overall system you’re developing
  - why different image size is a problem:
    - we feed images in batches -> tensor (fails, has to be same size)
    - batches -> transform -> tensor
    - squish/stretch leads to unrealistic shape, cropping removes important features, best way: randomly select different parts of an image and crop
  - error analysis:
    - sort images by loss (high to low) -> debug where errors are occuring (due to mislabelling, bad data, or model problem)
    - loss is high when : confident of incorrect answer / unconfident of correct answer
  - deployment issues
    - out-of-domain data (data drift) : data used to train model is very different from production
    - domain shift: type of data model sees changes over time (insurance company represent different risk over time, model no longer relevant)
  - solution
    - manual process: human checking model output
    - limited scope deployment: supervised and deployed on a trial basis
    - gradual expansion: implement good reporting systems and consider what could go wrong
  - thought experiment for rolling out ML systems
    - “What would happen if it went really, really well?”
    - who would be most impacted? What would the most extreme results potentially look like? How would you know what was really going on?
    - help you to construct a more careful rollout plan, with ongoing monitoring systems and human oversight

Links 🔗

- [2. From Model to Production | Deep Learning for Coders with fastai and PyTorch](https://learning.oreilly.com/library/view/deep-learning-for/9781492045519/ch02.html#idm46668598893488)

## Day 139: Jan 8, 2023

- worked on bird/plane model with fastai

Links 🔗

- [is it a bird or a plane](https://www.kaggle.com/code/benthecoder/is-it-a-bird-or-a-plane)

## Day 140: Jan 9, 2023

- ChatGPT for seo
  - Keyword research : "Keyword ideas for TOPIC"
  - Ideas: "10 blog ideas for TITLE"
  - outlines: "Blog outlines for: TOPIC"
  - titles: "Write an SEO optimized title that will improve click through rate in a playful tone for: TITLE"
  - semantic keywords: "Semantic keywords for: TOPIC"

Links 🔗

- [ChatGPT to 10x SEO](https://twitter.com/barrettjoneill/status/1610629309236150272?s=48&t=j92cl6mXt2Vb2cVtBOBo2w)
- [f/awesome-chatgpt-prompts: This repo includes ChatGPT promt curation to use ChatGPT better.](https://github.com/f/awesome-chatgpt-prompts)
- [Why SEO Pros Need To Master Prompts: The ChatGPT Revolution](https://www.searchenginejournal.com/why-seo-pros-need-to-master-prompts-the-chatgpt-revolution/473780/#close)

## Day 141: Jan 10, 2023

- extreme questions
  - "If you were forced to increase your prices by 10x, what would you have to do to justify it?"
  - "If all our customers vanished, and we had to earn our growth and brand from scratch, what would we do?"
  - "If you were never allowed to provide tech support, in any form, what would have to change?"
  - "What would be the most fun thing to build?"
  - "If our biggest competitor copied every single feature we have, how would we still win?"
  - "What if we are forced to ship a full, completed (at least MVP) new feature, in just two weeks, that would delight and surprise some fraction of our customers."
  - "What if you were forced to charge customers in a completely different manner?"
  - "If you were not allowed to have a website, how would you still grow your business?"
  - "What if you made your most introverted teammates’ dreams come true: No more synchronous meetings, ever again?"
  - "If we could never talk to our customers again, how would we figure out what to build?"
  - "What if it didn’t matter how unprofitable you were?"
  - What if you could change anything, regardless of what anyone thinks or feels?"
  - "What externality has the potential to kill the entire company?"
  - "What if our only goal was to create the most good in the world, personally for our customers?"
  - "What if you could only ship one thing this year?"

Links 🔗

- [Extreme questions to trigger new, better ideas | A Smart Bear: Longform](https://longform.asmartbear.com/posts/extreme-questions/)

## Day 142: Jan 11, 2023

- happiness habits
  - \>7 hours of sleep
  - Personal hobby (art, writing, music, cooking, reading, gaming)
  - exercise / sports
  - spend time in nature (marvel at flowers, plants, birds, insects.)
  - meditate
  - pray
  - spend time with friends outside of office/professional setting
  - engage with support groups or therapist
  - spend time with family outside of household
- cope on bad days and dealing with stress
  - comedic movies
  - reading poetry
  - breathing exercises
  - revisit favorite book
  - listen to upbeat songs
  - journaling
  - talk to a friend
- social relationships = building blocks of happiness.
  - We all stand to benefit from close friendships, romantic partners, and a “general sense of respect and belonging in a community,”
- meaning of happiness? subjective, but has these ingredients:
  - a sense of control and autonomy over one’s life
  - being guided by meaning and purpose
  - connecting with others
- happiness can be measured, strengthened, and taught
  - "The more you notice how happy or how grateful you are, the more it grows,"
- "Money can’t buy happiness"
  - BUT it can buy many things that contribute mightily: such as exciting experiences.
  - Spending money on others is also linked to happiness

Links 🔗

- [The Daily Habits of Happiness Experts | TIME](https://time.com/6241099/daily-habits-happiness-experts/)
- [The Happiness Revival Guide | TIME](https://time.com/collection/happiness-revival-guide/)
- [The Computer Science Book: a complete introduction to computer science in one book](https://thecomputersciencebook.com/)

## Day 143: Jan 12, 2023

- context switching
  - "most people average only 3 minutes on any given task before switching to something else (and only 2 minutes on a digital tool before moving on)."
  - Taking on additional tasks simultaneously can destroy up to 80% of your productive time:
    - Focusing on one task at a time = 100% of your productive time available.
    - Juggling two tasks at a time = 40% of your productive time for each and 20% lost to context switching.
    - Juggling three tasks at a time = 20% of your productive time for each and 40% lost to context switching.
  - 5 ways
    - (1) Batch and timeblock your schedule to create clearer ‘focus boundaries
    - (2) Build a habit of single-tasking throughout the day
      - Remove as many distractions as possible
      - start small and set a timer
      - Get rid of the ‘drains and incompletions’ that compete for your attention
    - (3) Add in routines and rituals that remove ‘attention residue’
      - “People need to stop thinking about one task in order to fully transition their attention and perform well on another. Yet, results indicate it is difficult for people to transition their attention away from an unfinished task and their subsequent task performance suffers.”
      - batch similar tasks together
      - build routines and rituals for when you need to ‘hard’ switch tasks
    - (4) Use regular breaks and rests to recharge
      - Do the 20/20/20 exercise to help reduce eye strain
      - Use breathing exercises to combat stress
      - Stretch, workout, or do a quick walk
      - Watch a funny video or something else relaxing
    - (5) Master the end-of-day shift from work to non-work mode
      - “If you strictly follow this after-work routine, you’ll soon discover that not only are you working harder when you work, but your time after work is more meaningful and restorative than ever before.”
      - strategies
        - Record your progress.
        - Organize any uncompleted tasks
        - glance at week ahead
        - acknowledge day is over

Links 🔗

- [Context switching: Why jumping between tasks is killing your productivity](https://blog.rescuetime.com/context-switching/)

## Day 144: Jan 13, 2023

- Make the most of our 24 hours
  - (1) be intentional at the start of each day
    - make a list of 5 things you want to do
  - (2) don’t shoot for doing more, do what matters
    - doing 30+ things in a day won’t get rid of the time scarcity — in fact, it often makes the stress even worse
  - (3) create moments of transcendence.
    - A moment of transcendence is something each of us has experienced: when we feel incredibly connected to the world around us, when we lose our sense of separate self and feel a part of something bigger (Flow state)
  - (4) reflect with gratitude
    - take a few moments to reflect back on your day and think about what you’re grateful for

Links 🔗

- [How to Make the Most of Your 24 Hours - zen habits zen habits](https://zenhabits.net/transcendent/)

## Day 145: Jan 14, 2023

- Forward-Forward algorithm (FF)
  - what: a technique for training neural networks that uses two forward passes of data through the network, instead of backpropagation, to update the model weights.
  - why: backprop requires full knowledge of the computation in the forward pass to compute derivatives and storing activation values during training
  - advantages
    - can be used when the precise details of the forward computation are unknown
    - can learn while pipelining sequential data through a neural network without ever storing the neural activities or stopping to propagate error derivatives
    - a model of learning in cortex and as a way of making use of very low-power analog hardware without resorting to reinforcement learning
  - how:
    - The first forward pass operates on positive data from a training set, and the network weights are adjusted to cause this input to increase a layer's goodness value
    - In the second forward pass, the network is given a generated negative example that is not taken from the dataset.
  - performance:
    - the FF-trained networks performed "only slightly worse" than those trained using backpropagation for CV task on MNIST and CIFAR datasets
    - Hinton’s paper proposed 2 different Forward-Forward algorithms, which I called Base and Recurrent. ... Base FF algorithm can be much more memory efficient than the classical backprop, with up to 45% memory savings for deep networks
  - 4 concepts
    - Local training. Each layer is trained just comparing the outputs for positive and negative streams
    - No need to store the activations. Activations are needed during the backpropagation to compute gradients, but often result in nasty Out of Memory errors.
    - Faster weights layer update. Once the output of a layer has been computed, the weights can be updated right away, i.e. no need to wait the full forward (and part of the backward) pass to be completed.
    - Alternative goodness metrics. Hinton’s paper uses the sum-square of the output as goodness metric, but I expect alternative metrics to pop up in scientific literature over the coming months.

Links 🔗

- [paper](https://www.cs.toronto.edu/~hinton/FFA13.pdf)
- [implementation](https://github.com/nebuly-ai/nebullvm/tree/main/apps/accelerate/forward_forward)
- [pytorch implementation](https://github.com/mohammadpz/pytorch_forward_forward)
- [twitter thread](https://twitter.com/diegofiori_/status/1605242573311709184)

## Day 146: Jan 15, 2023

- Polars
  - a blazingly fast DataFrame library written in Rust on top of Apache Arrow
  - CPU's today's come with many cores and with their superscalar designs and SIMD registers allow for even more parallelism. Polars is written from the ground up to fully utilize the CPU's of this generation
  - Besides blazingly fast algorithms, cache efficient memory layout and multi-threading, it consist of a lazy query engine, allowing Polars to do several optimizations that may improve query time and memory usage.

Links 🔗

- [GitHub](https://github.com/pola-rs/polars)
- [Introduction - Polars](https://pola-rs.github.io/polars-book/user-guide/)
- [Modern Polars](https://kevinheavey.github.io/modern-polars/?ref=blef-fr)
- [Polars: Blazingly Fast DataFrames in Rust and Python - YouTube](https://www.youtube.com/watch?v=kVy3-gMdViM)
- [Calmcode - polars: Introduction](https://calmcode.io/polars/introduction.html)
- [I wrote one of the fastest DataFrame libraries | Ritchie Vink](https://www.ritchievink.com/blog/2021/02/28/i-wrote-one-of-the-fastest-dataframe-libraries/)

## Day 147: Jan 16, 2023

- 4000 weeks
  - The average human life is only four thousand weeks (~80 yo)
  - The pressure to be more productive and fit ever-increasing quantities of activity into a stubbornly non-increasing quantity of time leads to productivity anxiety, shriveled attention spans, and burn-out.
  - productivity is a trap
    - The day will never arrive when you finally have everything under control
    - "Time feels like an unstoppable conveyer belt, bringing us new tasks as fast as we can dispatch the old ones; and becoming “more productive” just seems to cause the belt to speed up."
  - adopt a limit-embracing attitude
    - it's irrational to feel troubled by an overwhelming to-do list, Surrender to the reality that things just take the time they take, and confront your limitations
  - how you spend your time is a choice
    - accept that there will always be too much to do
    - understand that you’re guaranteed to miss out on almost every experience the world has to offer
    - you'll get to focus on fully enjoying the tiny slice of experiences you actually do have time for
  - importance of rest
    - Its absence can lead to stress, burnout, and counterintuitively overall poor performance despite the extra hours worked
    - Leisure is not merely an opportunity for recovery and replenishment for the purposes of further work, but for its intrinsic satisfactions
- 10 tools for embracing finitude
  - (1) Adopt a fixed volume approach to productivity
    - Establish pre-determined time boundaries on your work, and make decisions in light of those limits.
    - ex: I need to finish this by 5:30, so I'm have 4 hours left and I should use it wisely and not waste it on other lower value tasks
  - (2) Serialize, Serialize, Serialize
    - Focus on one big project at a time, and see it to completion before moving onto the next.
    - ex: i shouldn't start two articles, a side project, and my assignment all at once, I won't make much progress. Instead, I should train myself to consciously postpone everything except the one most important thing
  - (3) Strategic underachievement
    - you can't except excellence in all areas of life, decide in advance what to fail at.
    - ex: I know I'm not good at experimental design, expect I won't get perfect grades in my exam
  - (4) celebrate wins
    - keep a "done" list which starts empty and fills up over the day, lower the bar for accomplishments, small wins accrue
    - ex: even though I didn't finish that article, I gathered resources for that article, read a couple interesting blog posts, learned something new, read a chapter of a book. note all this down
  - (5) consolidate care
    - Consciously choose your battles in industry, charity, activism, and politics.
    - ex: focus on caring what matters the most: having good conversations, developing relationships w God and important people in my life, making real-impact to the world, care less about what others think, and monetary things
  - (6) Embrace boring & single-purpose technology
    - make your devices boring, remove distracting apps, choose single-purpose devices
    - ex: my ipad is for notes and reading, no distracting apps, I have bedtime mode and grayscale on after 10
  - (7) seek novelty in mundane
    - pay more attention to every moment no matter how mundane, whatevery draws your attention more fully to the present
    - ex: practice meditating, go on spontaneous walks, take pictures, journal everyday, be more mindful
  - (8) Be a researcher in relationships
    - when talking with someone, adopt an attitude of curiosity, where your goal is to figure out who this human being is
    - ex: next time when I'm striking up convo with a stranger, imagine I have to write a paper about them
  - (9) Cultivate instantaneous generosity
    - act on generous impulse right away, we are social creatures, generosity makes us happy
    - ex: give compliments when I think of them, check in on friends, praise someone, thank others.
  - (10) Practice doing nothing
    - "If you can’t bear the discomfort of not acting you’re far more likely to make poor choices with your time simply to feel as if you’re acting"
    - ex: set a 5min timer, sit down and do nothing, notice you do something (thinking, focusing on breahth, self-criticism), STOP IT.

Links 🔗

- [Four Thousand Weeks](https://leebyron.com/4000/)

## Day 148: Jan 17, 2023

- data science interview tips
  - (1) Master communication and storytelling
    - "You may be the strongest technical candidate, but you need to clearly articulate why you made certain decisions, identify caveats and limitations, and be concise in your answers, especially when you talk about the impact and route to the value of the work completed"
    - write down the context of the things you highlighted on your resume
      - why you picked this solution over another
      - what the impact or outcome was
      - whether or not the business implemented your solution
    - change the level of detail, going from executive and high-level summaries to detailed technical deep dives
  - (2) Understand the fundamentals of the company
    - invest some time to understand the company you are interviewing for,
      - primary purpose and positioning in the market
      - source of revenue
      - significant competitors
    - how?
      - visit corporate website: get a feel for the brand and what it stands for
      - Look for some financial information through the investor relations section or Yahoo Finance
      - Reach out to employees, convey your interest in the firm, and ask for 15 minutes of their time
    - two more tips
      - Demonstrate you have a keen sense for identifying pragmatic solutions and a strong product sense (a strong focus on the impact and outcome of the work rather than the work itself)
      - Be your authentic self, and don’t assume anything in the conversation
  - "Your preparation will set you apart from other candidates, especially in how you communicate, tell your story, and demonstrate your grasp of the company."

Links 🔗

- [What I've learned from interviewing more than 300 Data Scientists](https://medium.com/@martinleitner_33020/what-ive-learned-from-interviewing-more-than-300-data-scientists-dc5426a6df9d)
- [What skill is a significant differentiator for a Data Scientist?](https://medium.com/@martinleitner_33020/what-skill-is-a-significant-differentiator-for-a-data-scientist-d0a4af725a86)

## Day 149: Jan 18, 2023

- Neuroscience of Achieving Your Goals
  - (1) The 85% Rule — you should set your goals so that you achieve them 85.13% of the time (failing 15% of the time despite your best efforts)
  - (2) Make A Plan — SMART goals, create a specific set of action steps that get right down to details about what success would look like
  - (3) Imagine The Worst — write down/talk about how bad it will be if you don't achieve your goals, it’s motivating to think about what it will be like if you fail.
  - (4) Outsmart Your Obstacles — identify potential obstacles ahead of time and then planning out how to defeat them (foreshadow failure)
  - (5) Procrastinate with Other Tasks — pre-task multitasking before a work session can actually help you generate adrenaline and get you into action
  - (6) Focus Your Eyes to Focus Your Mind — Focus your visual attention on a single point for 30-60 seconds, it increases levels of systolic blood pressure and adrenaline
  - (7) Have A Weekly Check In — review the progress you’ve made on your goals once a week, “dopamine milestone” — a pit stop where you signal to your brain that things are moving forward
  - (8) Reward Your Effort — the next time you make an effort on something important — take note, and then take pride. Celebrate progress, not perfection, pat yourself on the brain

Links 🔗

- [The Neuroscience of Achieving Your Goals - Superorganizers - Every](https://every.to/superorganizers/the-neuroscience-of-achieving-your-goals)

## Day 150: Jan 19, 2023

- blogging lessons
  - (1) writing output == input
    - if writing as a way to organize our thoughts, then there should be thoughts to begin with
    - writing is thinking, we collate these thoughts from different inputs, and transform them into something uniquely our own
    - have a backlog of ideas and curate content you consume from internet, and take notes
  - (2) Optimize for readers who skim
    - Ask yourself, would a reader get what you're trying to say if they just skimmed through your post
    - tell a story through headings: headings should at least form a sentence when read from start to end
    - include a lot of illustrations deliberately
  - (3) views are nice, but they're not everything
    - figure out why you're writing, if it's to trace your DS journey, see yourself grow, views don't measure that
    - write fewer begineer posts
    - “two [blog posts] for me and one for them [the audience].”
    - endow your perspective when writing beginner articles, there could be a good analogy or compelling project, your views and take on it make it unique
  - (4) Blogging is half marketing
    - marketing starts before writing - search for keywords to use to title posts
    - Rule for good topic: if you labor over a concept/problem, and resources on internet are scarce
  - (5) Write a first draft when I’m inspired
    - inspiration has a half-life, write a first draft when it's peaking
    - pre-writing -> editing mode -> (if not interesting) discard
    - pre-writing: have materials and references ready, workshop a title, draft headings, write key takeaways, then expand these into bullet points
    - discarding drafts that do not spark joy
  - (6) Keep a changelog for every blog post
    - informs readers on how the blog post evolved, with all its corrections and feedback
    - blogs are also a record of your thoughts, past and present
    - blogpost annealing: apply heat (like metal) in form of feedback on blog post to shape it in a better form

Links 🔗

- [Lessons from six years of blogging](https://ljvmiranda921.github.io/life/2023/01/07/six-years/)
- [Blogpost Annealing](https://www.swyx.io/blogpost-annealing)

## Day 151: Jan 20, 2023

- How to understand things
  - 'intelligence' is as much about virtues such as honesty, integrity, and bravery, as it is about 'raw intellect’
  - Intelligent people simply aren’t willing to accept answers that they don’t understand — no matter how many other people try to convince them of it, or how many other people believe it, if they aren’t able to convince them selves of it, they won’t accept it.
  - (1) energy and intrinsic motivation: thinking hard takes effort
    - It’s easy to think that you understand something, when you actually don’t
    - to test understanding: attack the thing from multiple angles and see if you understand it
    - This requires a lot of intrinsic motivation, because it’s so hard; so most people simply don’t do it.
    - You have the drive, the will to know : not understanding something — or having a bug in your thinking — bothers you a lot
  - honesty, or integrity: a sort of compulsive unwillingness, or inability, to lie to yourself.
    - "first rule of science is that you do not fool yourself" - Feynman
  - (2) unafraid to look stupid
    - looking stupid takes courage, and sometimes it’s easier to just let things slide
  - Go slow
    - Read slowly, think slowly, really spend time pondering the thing.
    - Start by thinking about the question yourself before reading a bunch of stuff about it.
    - A week or a month of continuous pondering about a question will get you surprisingly far.
  - understanding is not a binary “yes/no”. It has layers of depth

Links 🔗

- [understanding - nabeelqu](https://nabeelqu.co/understanding)
- [Noticing Confusion - Sequence](https://www.readthesequences.com/Noticing-Confusion-Sequence)

## Day 152: Jan 21, 2023

- OpenAI Embeddings
  - what:
    - vector (list) of floating point numbers
    - The distance between two vectors measures their relatedness. (closer = high relatedness and vice versa)
  - use cases:
    - Search (where results are ranked by relevance to a query string)
    - Clustering (where text strings are grouped by similarity)
    - Recommendations (where items with related text strings are recommended)
    - Anomaly detection (where outliers with little relatedness are identified)
    - Diversity measurement (where similarity distributions are analyzed)
    - Classification (where text strings are classified by their most similar label)
  - embedding models
    - second generation: `text-embedding-ada-002`, uses the `cl100k_base` tokenizer, has `8191` max input tokens
  - limitation and risks
    - social bias:
      - evidence of bias from SEAT (May et al, 2019) and the Winogender (Rudinger et al, 2018) benchmarks
      - models more strongly associate (a) European American names with positive sentiment, when compared to African American names, and (b) negative stereotypes with black women
    - english only
      - perform poorly on dialects or uses of English that are not well represented on the Internet
    - blind to recent events
      - contain some information about real world events up until 8/2020. If you rely on the models representing recent events, then they may not perform well.
  - FAQ
    - how to get number of tokens
      - second gen models no way to count locally, need call API first
      - first gen, use [OpenAI tokenizer page](https://beta.openai.com/tokenizer) or [transformers.GPT2TokenizerFast](https://huggingface.co/docs/transformers/model_doc/gpt2#transformers.GPT2TokenizerFast)
    - how to retrive K nearest embeddings vectors quickly
      - [Pinecone](https://www.pinecone.io/), a fully managed vector database
      - [Weaviate](https://weaviate.io/), an open-source vector search engine
      - [Faiss](https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/), a vector search algorithm by Facebook
  - code
    - [amazon food reviews](https://github.com/openai/openai-cookbook/blob/main/examples/Obtain_dataset.ipynb)
    - [visualize embeddings 2d](https://github.com/openai/openai-cookbook/blob/main/examples/Visualizing_embeddings_in_2D.ipynb)
    - [regression](https://github.com/openai/openai-cookbook/blob/main/examples/Regression_using_embeddings.ipynb)
    - [classification](https://github.com/openai/openai-cookbook/blob/main/examples/Classification_using_embeddings.ipynb)
    - [zero shot classification](https://github.com/openai/openai-cookbook/blob/main/examples/Zero-shot_classification_with_embeddings.ipynb)
    - [user and product embeddings](https://github.com/openai/openai-cookbook/blob/main/examples/User_and_product_embeddings.ipynb)
    - [clustering](https://github.com/openai/openai-cookbook/blob/main/examples/Clustering.ipynb)
    - [semantic text search](https://github.com/openai/openai-cookbook/blob/main/examples/Semantic_text_search_using_embeddings.ipynb)
    - [code search](https://github.com/openai/openai-cookbook/blob/main/examples/Code_search.ipynb)
    - [recommendations](https://github.com/openai/openai-cookbook/blob/main/examples/Recommendation_using_embeddings.ipynb)
    - [question answering](https://github.com/openai/openai-cookbook/blob/main/examples/Question_answering_using_embeddings.ipynb)

Links 🔗

- [Introducing Text and Code Embeddings](https://openai.com/blog/introducing-text-and-code-embeddings/)
- [OpenAI API - examples](https://beta.openai.com/docs/guides/embeddings/what-are-embeddings)
- [paper](https://ar5iv.labs.arxiv.org/html/2201.10005)
- [code examples](https://github.com/openai/openai-cookbook/tree/main/examples)
- [namuan/dr-doc-search: Converse with book - Built with GPT-3](https://github.com/namuan/dr-doc-search)
- [hwchase17/langchain: ⚡ Building applications with LLMs through composability ⚡](https://github.com/hwchase17/langchain)
- [jerryjliu/gpt_index: An index created by GPT to organize external information and answer queries!](https://github.com/jerryjliu/gpt_index)

## Day 153: Jan 22, 2023

- Used Pycaret for a simple regression problem, some notes
  - i had issues with installation of lightgbm, needed to do brew install cmake and brew install and create a new conda environment
  - easy to use, really is automated machine learning, run setup with preprocessing parameters, run compare_models to train multiple algorithms, you have options to tune and create specific models with cross validation, and then stack or blend models, etc.
  - It also has mlflow experiment logging built in
  - only downside is it's pretty slow, but it's understandable since it's training so many algorithms and I'm on my laptop
  - only tried regression, can be used for classification, time series, topic modelling and more.

Links 🔗

- [website](https://pycaret.org/)
- [docs](https://pycaret.gitbook.io/docs/)
- [Tutorials](https://pycaret.gitbook.io/docs/get-started/tutorials)
- [github](https://github.com/pycaret/pycaret)

## Day 154: Jan 23, 2023

- learned about hadoop ecosystem and got some data engineering resources
  - Hadoop: A framework for storing and processing large amounts of data across a cluster of computers.
  - HDFS: The Hadoop Distributed File System for storing data.
  - MapReduce: A programming model for processing data in parallel.
  - Pig: A platform for analyzing large data sets using a high-level language.
  - Hive: A data warehousing tool for querying and managing large data sets stored in HDFS.

Links 🔗

- [How to learn data engineering](https://www.blef.fr/learn-data-engineering/)
- [Hadoop and Big Data - hadoop](https://jheck.gitbook.io/hadoop/)

## Day 155: Jan 24, 2023

- normalizing constant in bayes rule
  - the idea is integrate P(θ|y) ha to be one because it is a probability, so we need p(y) to make the numerator one. p(y) is our normalizing constant
  - if it’s missing, we need simulation to find what p(y) is, ex: MCMC
- integration is a weighted average
  - the weighted aspect is the probability (height of the curve)
  - the average is dividing by the number of points (slices) we are integrating over

Links 🔗

- [probability - Normalizing constant in Bayes theorem - Cross Validated](https://stats.stackexchange.com/questions/12112/normalizing-constant-in-bayes-theorem)
- [Bayes' Theorem](https://chrispiech.github.io/probabilityForComputerScientists/en/part1/bayes_theorem/)

## Day 156: Jan 25, 2023

- why Bayesian?
  - theoretical reason
    - if p(𝛉) and p(y|𝛉) represents a rational person's belief, Bayesian is the optimal way to update that
    - optimal here = minimizes the expected loss
    - Bayesian estimates are > frequentist estimates
    - classical estimates of SE and CIs are overly optimistic because it doesn't account for all sources of uncertainty in the parameters
    - ex: mixed moels
      - 1. get estimates of variance components (VCs) of fixed and random effects
      - want to estimate BLUEs and BLUPs
      - proceed as VCs are known -> plug in
      - estimates of BLUPs are not accurate because you're acting as if you knew the VCs
  - practical reasons
    - choosing good P(θ) can be difficult and ad-hoc choices are used
    - even so, P(𝛉) leads to useful P(θ|y) given enough data (once n is large enough, effects of priors are negligible)
    - intepreration is more useful
  - in summary
    - bayesian: use probability to describe beliefs about parameters incorporating prior info and conditioning on the data
    - frequentist: treats parameters as unknown fixed quantities, base exclusively on the data and averages over the sample (randomly sample data -> calculations)

Links 🔗

- [Regression: What’s it all about? [Bayesian and otherwise] | Statistical Modeling, Causal Inference, and Social Science](https://statmodeling.stat.columbia.edu/2015/03/29/bayesian-frequentist-regression-methods/)
- [Bayesian vs Frequentist A/B Testing (and Does it Even Matter?)](https://cxl.com/blog/bayesian-frequentist-ab-testing/)
- [Statistical Alignment: Bayesian or Frequentist? | Built In](https://builtin.com/data-science/frequentist-vs-bayesian)
- [Bayesians are frequentists | Statistical Modeling, Causal Inference, and Social Science](https://statmodeling.stat.columbia.edu/2018/06/17/bayesians-are-frequentists/)
- [Bayes.ppt](https://www.cse.psu.edu/~rtc12/CSE586/lectures/BayesianEstimation.pdf)

## Day 156: Jan 26, 2023

- sql window functions
  - to get particular rank, do rank in subquery, and in outer query do where rank = N
  - to do rolling averages, create CTE with counts, then do partition and order by and the key terms PRECEDING, CURRENT ROW, FOLLOWING, and UNBOUNDED, etc. to get sliding window. To get 3-day i.e. is `ROWS BETWEEN 2 PRECEDING AND CURRENT ROW`
- tips for prompt engineering
  - Good results won’t come ASAP, it takes me 6-12 attempts to get a prompt to a point where I think it’s decent. Don’t expect magic on the first try.
  - Iterate fast, and don’t get too attached to specific prompts you’re working on.
  - If your output doesn’t end up looking as good as mine, that’s fine. GPT-3’s output can vary wildly based on a few words even.
  - Don’t get sucked in trying to make one specific prompt work. After 6-12 attempts, feel free to say fuck it and either a) start the prompt from scratch and change it up a ton or b) move on to the next lesson.
- Temperature
  - Temperature is basically how risky you want the model to be.
  - A higher temperature = model will be a bit more unhinged and make more mistakes.
  - A lower temperature means the model will take less risks and just give the more standard “right answer”.
- basics of GPT-3
  - made up of lots of parameters, each parameter is a specialized number
  - input sentence -> combine sentence with parameters -> prediction
  - 175 billion parameters (800 GB)
  - each parameter looks like -> `output = input * parameter`
  - sentences -> tokens through embeddings that are essentially a dictionary created that maps pieces of words to numbers
  - GPT-3 understands language as a collection of numbers and relationship those numbers have with parameters
  - parameters
    - trained on the entire dataset: wikipedia, books, news, etc on nearly 500 billion tokens
  - it costs about $5M in GPU time to train GPT-3.

Links 🔗

- [Build your own AI writing assistant w/ GPT-3](https://buildspace.so/builds/ai-writer)

## Day 157: Jan 27, 2023

- the Big Mac Index
  - bi-annual update by The Economist
  - telling you how much a McDonalds burger will cost on your winter vacation
  - lighthearted test of the economic theory of purchasing power parity (PPP)
    - the idea that exchange rates should settle at a place where identical goods and services cost the same in every country.
  - this is a rudimentary means of testing the theory, using only the Big Mac’s ingredients as the “basket of goods” rather than the hundreds used in more comprehensive PPP studies
  - If you’re an FX trader who believes in the theory of PPP, the index gives you plenty of ideas to work with

Links 🔗

- [Our Big Mac index shows how burger prices are changing | The Economist](https://www.economist.com/big-mac-index?utm_source=chartr&utm_medium=newsletter&utm_campaign=chartr_20230127)

## Day 158: Jan 28, 2023

- `EXPLAIN` - cost of the query(“cost units are arbitrary, but conventionally mean disk page fetches”)
- `EXPLAIN ANALYZE` - cost to execute the query and get the actual time taken as well
- flush DNS cache - `sudo dscacheutil -flushcache; sudo killall -HUP mDNSResponder`

Links 🔗

- [PostgreSQL: Documentation: 9.3: EXPLAIN](https://www.postgresql.org/docs/9.3/sql-explain.html)
- [6 Key Concepts, to Master Window Functions · Start Data Engineering](https://www.startdataengineering.com/post/6-concepts-to-clearly-understand-window-functions/)

## Day 159: Jan 29, 2023

- `MOD()` or `%` - to get remainder of division in SQL
- `SUM(CASE WHEN <CONDITION> THEN <COLUMN_TO_SUM> ELSE 0 END)` - to get sum of column that meet condition
- `date_part` is same as extract, but `date_part` is for postgres while extract is for standard SQL

Links 🔗

- [PostgreSQL: Documentation: 15: 9.9. Date/Time Functions and Operators](https://www.postgresql.org/docs/current/functions-datetime.html)

## Day 160: Jan 30, 2023

- black and white movies absolutely worth watching
  - Seven Samurai, original 1954 version
  - Casablanca
  - Yojimbo
  - Sanjuro
  - Double Indemnity
  - Sunset Blvd.
  - Witness for the Prosecution
  - Some Like It Hot
  - Twelve Angry Men (1957)
  - The Man From Earth (2007)
  - Young Frankenstein
  - To Kill a Mockingbird
  - Arsenic and old lace
  - Harvey (1950)
  - All About Eve
  - The Night of the Hunter
  - Roman Holiday
  - Gaslight
  - Metropolis
  - Night of the living dead

Links 🔗

- [Which black and white movies are absolutely worth watching? : AskReddit](https://www.reddit.com/r/AskReddit/comments/10opflg/which_black_and_white_movies_are_absolutely_worth/)

## Day 161: Jan 31, 2023

- performance measures for algos (4 ways)
  - Completeness; Does it always find a solution if one
    exists?
  - Optimality; Does it always find a least-cost solution?
  - Time Complexity; Number of nodes generated?
  - Space Complexity; Number of nodes stored in memory during search?
- Time and space complexity are measured in terms of problem difficulty - size of state space - defined by:
  - b – (maximum) branching factor of the search tree
  - d - depth of the shallowest optimal solution
  - m - maximum length of the state space

Links 🔗

- [Search Control](https://www.cs.miami.edu/home/geoff/Courses/COMP6210-10M/Content/SearchControl.shtml)

## Day 162: Feb 1, 2023

- UDFs
  - User-defined functions (UDFs) are custom functions defined by a user to allow for the reuse of custom logic in an environment. UDFs can be written in Python and used when writing Spark SQL queries, or they can be designed to compute on each row or groups of rows
- stateless hashing
  - Stateless hashing is a type of hash-based cryptography that does not require the use of a stateful signature scheme.
  - It is based on the eXtended Merkle Signature Scheme (XMSS) and uses a tree-of-trees structure with a number of few time signatures at the bottom.
  - This allows it to be used as a drop-in replacement for current signature schemes.
  - Stateful hash-based cryptography, on the other hand, uses a single Merkle public key to hash one-time public keys.
- bourgeois vs managerial capitalism
  - bourgeois: owner and manager of company (ex: a16z)
  - manageriol: owner is separate from manager, managers don't create new things, people leave companies, create startup under bourgeois capitalism
- data semantics
  - A semantic layer is a business representation of data that provides a consistent way of interpreting data and enables end-users to quickly discover and access data using standard search terms. It maps complex data into familiar terms and offers a unified and consolidated view of data across an organization
- sufficient statistic
  - A sufficient statistic is a statistic that summarizes all of the information in a sample about a chosen parameter. It is said to be sufficient for a parameter if the conditional distribution of the sample given a value of the statistic does not depend on the parameter. Intuitively, this means that no other statistic can provide more information about the parameter than the sufficient statistic[. Examples include the sample mean, which estimates the population mean, and T, which is said to be sufficient if the statistician who knows its value can do just as well as someone who knows all of the data points.
- fanout join
  - Fanout join can be used in SQL to LEFT JOIN two data models with a one-to-many relationship. This can cause issues when trying to aggregate data, such as when using SUM with a fanout on a join. To avoid this, it is important to define primary keys correctly and use the COALESCE function to see zeros instead of nulls. Additionally, fan traps can arise from the fundamental properties of SQL joins, so it is important to be aware of these potential issues.
- variance reduction
  - Variance reduction is a technique used in Monte Carlo methods to increase the precision of estimates obtained from a given simulation or computational effort. It is achieved by reducing the variance of the estimator, which is the measure of how far an estimate deviates from its expected value. This can be done by either increasing the number of simulations (i.e. running the simulation for a longer time) or by using variance reduction techniques such as splitting and Russian roulette, with ant colony methods as builders of importance maps. By reducing the variance, more accurate estimates can be obtained with fewer simulations, thus saving time and resources
  - Variance reduction techniques can be used to improve the power of A/B tests by attempting to reduce the variability of the results. This can be done by computing the covariance, variance, and mean of X and Y, as well as using methods such as CUPED (Covariate-Uniformly Partitioned Experiment Design)[
- sequential testing
  - Sequential testing is a practice of making decisions during an A/B test by sequentially monitoring the data as it accrues.
  - It is a type of statistical analysis where the sample size is not fixed in advance, and involves calculating test statistics using a specific type of sequential test called mixture Sequential Probability Ratio Test (SPRT).
  - This can lead to three different outcomes: reject the null hypothesis, accept the null hypothesis, or continue collecting data

Links 🔗

- [scientism on Twitter: "The tech billionaire crowd represent a kind of 'trad capitalism' that valorizes the owner-manager. They imagine themselves like the great industrialists and oppose themselves to modern capital where owners are divorced from management by dozens of layers of financial abstraction." / Twitter](https://twitter.com/mr_scientism/status/1519041920806395907)
- [Experimentation Platform in a Day | by Dan Frank | Deliberate Data Science | Medium](https://medium.com/deliberate-data-science/experimentation-platform-in-a-day-c60646ef1a2)
- [Data semantics: the missing layer of your data warehouse - Blog](https://www.coinbase.com/blog/data-semantics-the-missing-layer-of-your-data-warehouse)
- [What is a Semantic Layer?  - Datameer](https://www.datameer.com/blog/what-is-a-semantic-layer/)
- [Increasing the sensitivity of A/B tests by utilizing the variance estimates of experimental units - Meta Research | Meta Research](https://research.facebook.com/blog/2020/10/increasing-the-sensitivity-of-a-b-tests-by-utilizing-the-variance-estimates-of-experimental-units/)

## Day 163: Feb 2, 2023

- 7 steps in AB testing
  - (1) problem statement: what is the goal?
    - clarify user journey (user funnel)
      - ex: user visit -> search -> browse -> view item -> purchase
    - success metric
      - measurable: can the behaviour be tracked?
      - attributable: can you establish the cause -> effect relationship?
      - sensitive: does it have low variability to distinguish trt from control?
      - timely: can you measure success behavior in a short term?
  - (2) hypothesis testing: what results do you hypothesize from experiment
    - state hypothesis statements
      - null: no difference between trt and control
      - alternative: there is a difference between trt and control
    - set significance level (alpha)
      - the decision threshold, commonly 0.05, tells us that what we observed wasn't due to chance alone
      - alpha = probability of rejecting null hypothesis when it is true (false positive)
      - interpretation of alpha = 0.05: if we run the experiment 100 times, we expect 5 of them to be false positives
    - set statistical power (1-beta)
      - beta: probability of accepting null hypothesis when it is false (false negative)
      - power: probability of detecting an effect given alternative hypothesis is true (true positive)
    - set minimum detectable effect (MDE)
      - practical significance: how much difference is meaningful to the business?
      - typically 1% lift
      - ex: revenue per day: if change is at least 1% in higher in revenue per day, then it is practically significant
  - (3) experiment design: what are exp parameters?
    - set randomization unit
      - user, cookie, page or session level, device, etc.
    - target population in experiment
      - where to target in the funnel, ex: users who are searching something
    - determine sample size
      - n = 16 \* $\sigma^2$ / $\delta^2$
        - $\sigma$ is sample standard deviation
        - $\delta$ is difference between control and treatment
    - duration?
      - typically 1-2 weeks (to prevent day of the week effect)
  - (4) run experiment: what are requirements?
    - set up instruments and data pipelines to collect data
    - avoid peeking p-values
      - increases risk of false positive
  - (5) validity checks: did experiment run as expected?
    - instrumentation effect: guardrail metrics (bugs, glitches)
    - external factors: holiday, competition, economic disruptions
    - selection bias: A/A test (distribution of control == treatment)
    - sample ratio mismatch: chi-square goodness of fit test
    - novelty effect: segment by new and old visitors
  - (6) interpret results: which direction is metric significant statistically and practically
    - at signficance level at 0.05, there is/isn't statistical significance to reject the H0 and conclude that \_\_\_\_
  - (7) launch decision: based on result and tradeoffs, should we launch or not?
    - metric trade-offs: primary metric may improve but secondary decline
    - cost of launching: launching and maintaining change might be costly
    - risk of false positive: negative consequence to user -> churn -> revenue loss

Links 🔗

- [A/B Testing in Data Science Interviews by a Google Data Scientist | DataInterview - YouTube](https://www.youtube.com/watch?v=DUNk4GPZ9bw)

## Day 164: Feb 3, 2023

- Python and transformer study guide

Links 🔗

- [Python’s “Disappointing” Superpowers - lukeplant.me.uk](https://lukeplant.me.uk/blog/posts/pythons-disappointing-superpowers/)
- [Understanding all of Python, through its builtins](https://sadh.life/post/builtins/)
- [dair-ai/Transformers-Recipe: 🧠 A study guide to learn about Transformers](https://github.com/dair-ai/Transformers-Recipe)

## Day 165: Feb 4, 2023

- 8 key data structure that power modern databases
  - skip list -> sorted set
    - allows for fast lookup, ranged queries
    - alt. to balance tree, efficient search and insertion and deletion
    - redis
  - hash index
    - efficiently map key to values
    - fast lookups, insertion and deletion
    - redis, MySQL, PostgreSQL
  - LSM Tree
    - LSM tree: SStable + Memtable
      - SStable + memtable work together to handle high volumne of write operations
      - SSTable: sorted string table
        - store data on disk in sorted order
        - file based DS to store large amounts table in highly compressed and efficient format
      - Memtable: in memory table DS taht stores recent writes
    - cassandra, RocksDB, leveldb
  - B-tree
    - efficiently store and retrive large amounts of data on disk
    - balance tree: each node can have multiple children and keeps data sorted
    - b+ tree: all data is stored in leaf nodes and internal nodes only hold keys
    - MYSql, PostgreSQL, Oracle
  - Inverted Index
    - efficiently search and retrieve large collections of text data
    - maps words to documents in which they appear
    - why inverted? maps words to docs rather than the other way round
    - elastic search
  - suffix tree
    - efficient text search
  - R-tree
    - spacial index DS that organizes data based on geometric boundaries
    - efficiently search and retrive spacial data
    - PostGIS, Mongodb, elastic search

Links 🔗

- [8 Key Data Structures That Power Modern Databases - YouTube](https://www.youtube.com/watch?v=W_v05d_2RTo)
- [The Secret Sauce Behind NoSQL: LSM Tree - YouTube](https://www.youtube.com/watch?v=I6jB0nM9SKU&t=0s)
- [Database Internals](https://www.databass.dev/)

## Day 166: Feb 5, 2023

- history of tech
  - Platform I: Creating the ‘Silicon Valley’ (60s Semiconductor boom)
    - 1940s: invention of transistors
    - 1954: first silicon transistor
    - 1957: tratorous eight -> fairchild semiconductor
    - 1960s: Intel, AMD, Applied materials (valley of silicon)
  - Platform II: Inventing the PC (70s and 80s PC boom)
    - 1975: MITS Altair 8800
    - 1976: Apple I
    - 1977: PC “trinity” – the Apple II, Commodore’s PET, and RadioShack’s TRS-80
    - Microsoft (1975), Oracle (1977), Adobe (1982)
  - Platform III: Creating the Internet Economy (90s Internet boom)
    - 90s: Amazon, Google, and PayPal
    - 1998: advent of the world wide web
    - 1999: dotcom bubble
    - 2004: 48% of dotcom companies
    - Tesla and Palantir (2003), Facebook (2004), and Palo Alto Networks (2005)
  - Platform IV: Mobile computing (00s mobile boom)
    - 2007: launch of iphone
    - Airbnb (2008), Uber (2009), Snapchat (2011)
  - Platformization Conjecture
    - Company success is more likely when founded to exploit a new leap of technology.
    - Company success is more likely when founded during a period of higher-than-average interest rates.
    - Company success is even more likely when companies are founded to exploit a technology innovation that involves both software and hardware during periods of higher-than-average interest rates.
  - Platform V: Energy and compute become free and abundant
    - The first is the marginal cost of energy going to zero. This happens through a proliferation of zero-carbon solutions like solar and wind, along with moderately intensive solutions like natural gas, all working to create energy that quickly approaches $0/kwh.
    - Second, complementing this new energy model is a shift away from Moore’s Law and CPUs to the proliferation of GPUs. This would support scaling Moore’s Law through parallelism, which favors applications of machine learning and AI. As a result, the marginal cost of compute will go to zero.

Links 🔗

- [Higher Rates Will Lead to the Next Generation of Great Tech Startups](https://chamathreads.substack.com/p/higher-rates-will-lead-to-the-next)
- [All-In with Chamath, Jason, Sacks & Friedberg - E114: Markets update: whipsaw macro picture, big tech, startup mass extinction event, VC reckoning](https://podcasts.google.com/feed/aHR0cHM6Ly9hbGxpbmNoYW1hdGhqYXNvbi5saWJzeW4uY29tL3Jzcw/episode/MzQ0MTJhZWYtMWJmNC00Yjk0LWIxMDEtYTU0ODA2YzAxMmZk?hl=en&ved=2ahUKEwjzut2pm__8AhXxl4kEHWhaCAsQjrkEegQICBAF&ep=6)

## Day 167: Feb 6, 2023

- how to choose priors
  - literature or scientific knowledge
  - elicit information from experts
  - choose prior that is mathematically convenient
  - difficulty: condense info in form of distribution
- conjugate prior
  - formal def: F a class of sampling distributions and P a class of prior distributions. Then P is conjugate for F if p(θ) ∈ P and p(y|θ) ∈ F implies that p(θ|y) ∈ P.
  - all likelihood from exponential family has conjugate
  - beta prior + binomial likelihood → Beta(a*, b*)
    - a\* = number of success + a
    - b\* = number of failures + b
    - a and b will get their own prior (hierarchical model)
  - conjugate because posterior is same form as prior
  - why nice? before you do anything, you know what posterior will look like
  - posterior mean is weighted average of prior mean and data mean
- non informative prior
  - A prior distribution for θ is non-informative if it is invariant to monotone transformations of θ.

Links 🔗

- [Bayesian method (1). The prior distribution | by Xichu Zhang | Towards Data Science](https://towardsdatascience.com/bayesian-method-1-1cbdb1e6b4)
- [Chapter 9 Considering Prior Distributions | An Introduction to Bayesian Reasoning and Methods](https://bookdown.org/kevin_davisross/bayesian-reasoning-and-methods/prior.html)
- [How Should You Think About Your Priors for a Bayesian Analysis? | Steven V. Miller](http://svmiller.com/blog/2021/02/thinking-about-your-priors-bayesian-analysis/)

## Day 168: Feb 7, 2023

- transformers resources and papers

Links 🔗

- [The Illustrated Transformer – Jay Alammar – Visualizing machine learning one concept at a time.](https://jalammar.github.io/illustrated-transformer/)
- [The Transformer Family | Lil'Log](https://lilianweng.github.io/posts/2020-04-07-the-transformer-family/)
- [Transformer models: an introduction and catalog — 2023 Edition - AI, software, tech, and people, not in that order… by X](https://amatriain.net/blog/transformer-models-an-introduction-and-catalog-2d1e9039f376/)
- [karpathy/nanoGPT: The simplest, fastest repository for training/finetuning medium-sized GPTs.](https://github.com/karpathy/nanoGPT)
- [Introduction to Deep Learning](https://sebastianraschka.com/blog/2021/dl-course.html#l19-self-attention-and-transformer-networks)
- [Understanding Large Language Models -- A Transformative Reading List](https://sebastianraschka.com/blog/2023/llm-reading-list.html)

## Day 167: Feb 8, 2023

- GP resources

Links 🔗

- [Gaussian Processes: a versatile data science method that packs infinite dimensions – @aurimas](https://aurimas.eu/blog/2022/03/intro-to-gaussian-processes/)
- [Implement A Gaussian Process From Scratch | by Shuai Guo | Towards Data Science](https://towardsdatascience.com/implement-a-gaussian-process-from-scratch-2a074a470bce)
- [Gaussian processes (1/3) - From scratch](https://peterroelants.github.io/posts/gaussian-process-tutorial/)
- [jwangjie/Gaussian-Processes-Regression-Tutorial: An Intuitive Tutorial to Gaussian Processes Regression](https://github.com/jwangjie/Gaussian-Processes-Regression-Tutorial)

## Day 168: Feb 9, 2023

- think fast, talk smart: communication techniques
  - Anxiety management
    - 85% of people are nervous when they speak in public. Anxiety isn't inherently a bad thing it can help you focus. However, excessive anxiety could impair our ability to speak spontaneously.
    - The techniques used in anxiety management:
      - When anxiety symptoms kick in few minutes before public speaking (as in gurgly stomach, shaking limbs, etc.), just be mindful about them, acknowledge them and don't resist them "We simply greet our anxiety and say hey" Take a deep breath and don't let anxiety spiral out of control.
      - Re-framing public speaking as a conversation and not a performance. There is no "right" or "wrong" way of presenting (although there are certainly better or worse ways). there are multiple ways to make it as a conversation like:
      - Start with questions: they are dialogic in nature. They could be rhetorical, polling, or simply asking for information.
      - Using conversational language. Using an inclusive language and not distance the audience from ourselves and the speech, in addition to having a relaxed body language.
      - Be present oriented. Don't think about the far future. This in turn will clear your mind and make you less nervous. There are some ways of becoming present in the moment such as doing pushups, walking, listening to music, tongue twisters (they can help in warming up the voice as an added benefit) or it could be anything that brings the attention and use some cognitive resources.
  - Ground rules for being comfortable in speaking in spontaneous situations
    - Get out of your own way. Dare to be dull. Don't worry about being perfect or doing stuff flawlessly. Improvise, don't stockpile information, let your brain act spontaneously. Train the skill of improvisation. Because aiming at greatness could be in your way due to over evaluation, and over analyzation which leads to freezing up.
    - See things as opportunities and not as challenges or threats. Make it a conversation and don't make it an adversarial situation. Make it an opportunity to clarify and explain what's in your head, and understand what people are thinking. Take a "Yes, and.." approach instead of "No, but..".
    - Slow down and listen. "You need to understand the demands of the requirement you find yourself in, in order to respond appropriately". Don't jump to conclusions without gathering enough information. So, slow down and listen to understand and be in touch with the receiver to fulfill your obligation as a communicator. "Don't just do something, stand there."
    - Tell a story. Respond in a structured way. Having a structure is key to having a successful spontaneous, and planned speaking. It increases processing fluency which is how effective we process information. We process and retain structured information 40% more reliably and accurately than non-structured ones. For example to memorize a string of 10 numbers we structure them into 3,3, and 4 numbers. Structure helps us Remember.
  - A couple of useful structures:
    - The "Problem > Solution > Benefit" Structure. You firstly start talking about what is the problem, then talk about a way/ways to solve the problem, and finally, talk about the benefit of solving it. Never lose your audience. Set expectations and provide a structure to keep the listener on track, and this structure helps with that. Could be re-framed as "Opportunity > Solution [steps to achieve it] > Benefit"
    - The "What? So what? Now what?" structure. Start firstly by talking about what the problem/idea is, why is it important, and then what the next steps are. This is a good formula for answering questions, and introducing people [Who they are? Why are they important? And what to do next with them (listening,drinking, etc..)]. In a spontaneous speaking situation we have to think about two things simultaneously; Figure out what to say and how to say it.
  - Practicing these structures is a key skill for effective spontaneous thinking. "Structure sets you free."

Links 🔗

- [Think Fast, Talk Smart: Communication Techniques - YouTube](https://www.youtube.com/watch?v=HAnw168huqA)

## Day 169: Feb 10, 2023

- halfway into Kafka project
  - stream : unbounded sequence of ordered, immutable data
  - stream processing: continual calculations performned on one or more streams
  - immutable data: data that cannot be changed
  - event: immutable fact about something that has happened in the past
  - broker: a single member server of kafka cluster
  - cluster: a group of one or more kafka brokers working together to satisfy kafka production and consumption
  - Node: a single computing instance (physical server in datacenter or virtual machine in cloud)
  - zookeper: used by kafka brokers to determine which broker is the leader of a given partition and topic, and track cluster membership and configuration for kafka
  - data partition: kaflka topics consist of one or more partitions. partition is a log which provides ordering guarantees for all data contained, they are chosen by hashing key values.

Links 🔗

- [📈 Stock Market Real-Time Data Analysis Using Kafka | End-To-End Data Engineering Project - YouTube](https://www.youtube.com/watch?v=KerNf0NANMo)

## Day 170: Feb 11, 2023

- ideas -> words
  - Writing about something, even something you know well, usually shows you that you didn't know it as well as you thought. Putting ideas into words is a severe test
  - It's not just having to commit your ideas to specific words that makes writing so exacting. The real test is reading what you've written. You have to pretend to be a neutral reader who knows nothing of what's in your head, only what you wrote
  - there were things I didn't consciously realize till I had to explain them
  - A great deal of knowledge is unconscious, and experts have if anything a higher proportion of unconscious knowledge than beginners.
  - Putting ideas into words doesn't have to mean writing, of course. You can also do it the old way, by talking. But in my experience, writing is the stricter test. You have to commit to a single, optimal sequence of words

Links 🔗

- [Putting Ideas into Words](http://www.paulgraham.com/words.html)

## Day 171: Feb 12, 2023

- assign method pandas

Links

- [Chaining in Pandas | //gardnmi](https://gardnmi.github.io/blog/jupyter/pandas/python/2021/08/22/chaining-in-pandas.html)
- [Professional Pandas: The Pandas Assign Method and Chaining](https://ponder.io/professional-pandas-the-pandas-assign-method-and-chaining/)

## Day 172: Feb 13, 2023

- investing in US as F1 student and taxe
  - For taxes, it is completely irrelevant where your broker is based. If you are tax resident of the US, you are taxed on it in the US, even if you have an account in Europe and vice versa.
  - Typically, as F1 student you will count as exempt individual for the first 5 calendar years in the US, but this can be extended. Therefore, you would continue to count as tax resident of your European home country (even if you don't spend much time there). If you try to open a brokerage account in the US, you would need to say that you are not a US person (but rather a non-resident for tax purposes), which may even make it difficult to open an account in the US (Robinhood, Charles Schwab, Fidelity will often refuse or transfer you to their "international" account version, which is not as good as a local account). The details are regulated by a tax treaty, but typically capital gains will only be taxed in your home country and the US will only tax dividends of US companies (typically at 15% if your home country has a tax treaty). See [this summary](https://www.expatfinance.us/united-states/double-taxation) for more details.
  - If you ever become tax resident of the US (after five years, if you get greencard / citizenship or if you change visa / start a job), you will be US person, i.e., resident for tax purposes. Then you would be taxed in the US and you will be subject to all the rules.
  - MOST IMPORTANTLY: Non-US ETFs or other funds count as PFIC under US tax rules, which leads to huge problems (see [this summary](https://www.expatfinance.us/united-states/us-foreign-taxation)). **If you expect to ever become US tax resident, DO NOT INVEST in non-US ETFs.** Any European ETF will lead to problems if you ever become US tax resident (in most cases, it will be best to sell, which will likely incur taxes in your home country or in the US).
  - You could open / keep a brokerage account in Europe, but if you ever become US tax resident, you will likely need to close the account. Moreover, European accounts will not allow you to buy US-ETFs [due to the EU MiFiD regulation](https://www.expatfinance.us/united-states/us-foreign-taxation), so if you expect to become US tax resident, it is much better to buy US-ETFs in a US-based account, as a European account would only allow you to buy European ETFs, which would cause problems once you become US tax resident.
  - My recommendation: Open an [Interactive Brokers Lite account](https://www.expatfinance.us/united-states/interactive-brokers-lite) using your US address as permanent residence, but then declare under tax rules that you are an F1 student,so that you are not a US tax resident. This will even allow you to fill in a W8-BEN, so that US tax on US dividends is automatically deducted based on the tax treaty of your home country. If you want to invest in ETFs, only purchase US-domiciled ETFs, such as the standard Vanguard products (e.g., VT, VTI, VEA, VWO etc.). This account also allows you to easily and cheaply [transfer USD/EUR internationally](https://www.expatfinance.us/general/ibkr-money-transfers) and best of all, you can deposit/withdraw in all major currencies without any transfer fees.
  - If you become US tax resident, it will be easy to just change your tax residence in the account and you would not need to worry about PFICs, because you are already compliant with US tax laws.
  - If you move back to your home country, you have two options: You can either try to keep the account open based on your US address (see [this discussion](https://www.expatfinance.us/united-states/us-foreign-taxation) and [here on reddit](https://www.reddit.com/r/ExpatFinance/comments/o2mzkv/not_bring_100_honest_about_a_us_address/)), which makes sense if you expect to return to the US soon. If you move back to Europe longterm, you can just open an Interactive Brokers account in Europe and transfer all your assets to a European Interactive Brokers account. Having US-ETFs in your portfolio will not cause any problems, you may only be restricted from buying more, but you can still have these ETFs in your portfolio and sell them any time later (like during retirement etc.). In Germany, [US-ETFs even come with a noticeable tax advantage](https://www.expatfinance.us/germany/taxation-of-us-etfs) (also discussed here [on reddit](https://www.reddit.com/r/Finanzen/comments/omqq78/fundierte_analyse_wie_man_13_deutsche_steuer_auf/)).

Links 🔗

- [Which country should I open an investing account as an international in the US with an F1 visa? : eupersonalfinance](https://www.reddit.com/r/eupersonalfinance/comments/po7imd/comment/j8f9eph/?context=3)

## Day 173: Feb 14, 2023

- interview cheatsheet

Links 🔗

- [Job Search & Interview Cheatsheet](https://benthecoder.notion.site/Job-Search-Interview-Cheatsheet-709acbb9f1cc45e4bcfd2e77ec20f3b0)

## Day 174: Feb 15, 2023

- finished Kafka mini project
- Programming Language - Python
  - [dpkp/kafka-python: Python client for Apache Kafka](https://github.com/dpkp/kafka-python)
  - [fsspec/s3fs: S3 Filesystem](https://github.com/fsspec/s3fs)
- Apache Kafka
- Amazon Web Service (AWS)
  - EC2 - host kafka server
  - S3 (Simple Storage Service) - store data
  - Glue Crawler - create schema for data in S3
  - Glue Catalog - store schema for data in S3
  - Athena - Query data in S3

Links 🔗

- [benthecoder/kafka-stock-market](https://github.com/benthecoder/kafka-stock-market)

## Day 175: Feb 16, 2023

- update docker compose on mac

```bash
    sudo rm /usr/local/lib/docker/cli-plugins/docker-compose
```

Install with brew

```bash
    brew install docker-compose
```

and symlink it

```bash
    mkdir -p ~/.docker/cli-plugins
    ln -sfn /opt/homebrew/opt/docker-compose/bin/docker-compose ~/.docker/cli-plugins/docker-compose
```

if you want to check where it was installed

```bash
    docker info --format '{{range .ClientInfo.Plugins}}{{if eq .Name "compose"}}{{.Path}}{{end}}{{end}}'
```

Links 🔗

- [Uninstall Docker Compose | Docker Documentation](https://docs.docker.com/compose/install/uninstall/)
- [Docker desktop mac wont update docker compose - Stack Overflow](https://stackoverflow.com/questions/72663581/docker-desktop-mac-wont-update-docker-compose)

## Day 176: Feb 17, 2023

- Search War
  - Google
    - Google Search = 93 percent of all search-driven traffic
    - Google's in house models
      - Imagen image generator
      - LaMDA conversation generator
      - MusicLM music generator
      - PaLM large language model.
    - An astronomer quickly pointed out that the system had misstated the accomplishments of the James Web Space Telescope. The tech press pounced, and Google promptly lost roughly 8 percent of its market value.
  - microsoft
    - bing = 3 percent of search-driven traffic.
    - bing, edge and teams will utilize ai chatbot that will respond to conversational queries, summarize answers from multiple web pages, and generate text for emails, essays, advice, and so on.
    - A layer called Prometheus is intended to filter out incorrect or inappropriate results
      - prompts:
        - “informative, visual, logical, and actionable” as well as “positive, interesting, entertaining, and engaging.”
        - avoid answers that are “vague, controversial, or off-topic,”
        - present them with logic that is “rigorous, intelligent, and defensible.”
        - It must search the web — up to three times per conversational turn — whenever a user seeks information.
  - baidu
    - Baidu manages 65 percent of China’s search-driven traffic but less than 1 percent worldwide.
    - Baidu announced its own chatbot, Wenxin Yiyan, based on ERNIE. The company expects to complete internal testing in March and deploy the system soon afterward.
  - The cost of enhancing Google Search with ChatGPT output would approach $36 billion a year (roughly 65 percent of Google Search’s annual profit)

Links 🔗

- [AI Titans Clash, Deepfaked Propaganda Spreads, Generative Models Resurrect Seinfeld and more](https://www.deeplearning.ai/the-batch/issue-184/)
- [Can ChatGPT-like Generative Models Guarantee Factual Accuracy? On the Mistakes of Microsoft's New Bing - DEV Community 👩‍💻👨‍💻](https://dev.to/ruochenzhao3/can-chatgpt-like-generative-models-guarantee-factual-accuracy-on-the-mistakes-of-microsofts-new-bing-111b)

## Day 177: Feb 18, 2023

- set new remote origin `git remote set-url origin git://new.url.here`

Links 🔗

- [How to remove remote origin from a Git repository - Stack Overflow](https://stackoverflow.com/questions/16330404/how-to-remove-remote-origin-from-a-git-repository)

## Day 178: Feb 19, 2023

```py
import multiprocessing
import time

# bar
def bar():
    for i in range(100):
        print "Tick"
        time.sleep(1)

if __name__ == '__main__':
    # Start bar as a process
    p = multiprocessing.Process(target=bar)
    p.start()

    # Wait for 10 seconds or until process finishes
    p.join(10)

    # If thread is still active
    if p.is_alive():
        print "running... let's kill it..."

        # Terminate - may not work if process is stuck for good
        p.terminate()
        # OR Kill - will work for sure, no chance for process to finish nicely however
        # p.kill()

        p.join()
```

Links 🔗

- [python - Timeout on a function call - Stack Overflow](https://stackoverflow.com/questions/492519/timeout-on-a-function-call)

## Day 179: Feb 20, 2023

- Christianity -> human rights

Links 🔗

- [Why You're Christian - David Perell](https://perell.com/essay/why-youre-christian/)

## Day 180: Feb 21, 2023

- capture output of Process object

```python
import multiprocessing

ret = {'foo': False}

def worker(queue):
    ret = queue.get()
    ret['foo'] = True
    queue.put(ret)

if __name__ == '__main__':
    queue = multiprocessing.Queue()
    queue.put(ret)
    p = multiprocessing.Process(target=worker, args=(queue,))
    p.start()
    p.join()
    print(queue.get())  # Prints {"foo": True}

```

Links 🔗

- [python - How to get the return value of a function passed to multiprocessing.Process? - Stack Overflow](https://stackoverflow.com/questions/10415028/how-to-get-the-return-value-of-a-function-passed-to-multiprocessing-process/37736655#37736655)

## Day 181: Feb 22, 2023

- tqdm example

```python
    with tqdm(
        total=len(files * len(algs)),
    ) as t:
        for file in files:
            for alg in algs:
                t.set_description(f"Running {alg} on {file}")
                result = subprocess.run(
                    ["python", "main.py", "--fPath", f"{dir}/{file}", "--alg", alg],
                    capture_output=True,
                )
                if result.stderr:
                    raise subprocess.CalledProcessError(
                        returncode=result.returncode,
                        cmd=result.args,
                        stderr=result.stderr,
                    )
                with open(f"{out}/part2.txt", "a") as f:
                    f.write(f"--Fpath {file} --alg {alg} \n")
                    f.write(f"{result.stdout.decode('utf-8')} \n")

                t.update(1)

```

Links 🔗

- [tqdm/tqdm: A Fast, Extensible Progress Bar for Python and CLI](https://github.com/tqdm/tqdm)
- [Understanding Subprocesses in Python - Earthly Blog](https://earthly.dev/blog/subprocesses-in-python/)

## Day 182: Feb 25, 2023

- Learned about the meaning behind derivatives, and how backprop is actually just 100 lines of code and neural networks are 50 lines of code

Links 🔗

- [benthecoder/neural-networks-from-scratch](https://github.com/benthecoder/neural-networks-from-scratch)
