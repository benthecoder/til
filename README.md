# 66DaysOfData

## Progress

|                           Course/Book                           |        Status         |
| :-------------------------------------------------------------: | :-------------------: |
| [Machine Learning Specialization](https://tinyurl.com/2kydp9km) |   Course 2, Week 2    |
|         [MLOps Zoomcamp](https://tinyurl.com/2eyfcvbq)          | Week 4: ML Deployment |
|    [Data Engineering Zoomcamp](https://tinyurl.com/2egmk5yx)    |   Week 2 : Airflow    |
|                      Statistics Done Wrong                      |       Chapter 1       |
|                           Hands on ML                           |       Chapter 4       |
|                          Fluent Python                          |       Chapter 5       |

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
- [Lecture 03: Troubleshooting & Testing (FSDL 2022) - YouTube](https://www.youtube.com/watch?v=RLemHNAO5Lw)

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
- [A/B testing and growth analytics at AirBnb, metrics store-Nick Handel-the data scientist show#037 - YouTube](https://www.youtube.com/watch?v=5AH0zToK0e4)
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

- [Refactoring A Data Science Project Part 1 - Abstraction and Composition - YouTube](https://www.youtube.com/watch?v=ka70COItN40)
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

## Day 17: 16 Sepp 2022

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

- [Lecture 06: Continual Learning (FSDL 2022) - YouTube](https://www.youtube.com/watch?v=nra0Tt3a-Oc)

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

- [Lecture 07: Foundation Models (FSDL 2022) - YouTube](https://www.youtube.com/watch?v=Rm11UeGwGgk&t=1087s)

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
- [Illustrated Guide to Transformers Neural Network: A step by step explanation - YouTube](https://www.youtube.com/watch?v=4Bdc55j80l8&t=207s)

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

- [Sam Altman : How to Build the Future - YouTube](https://www.youtube.com/watch?v=sYMqVwsewSg)
- [Apache Kafka - IBM](https://www.ibm.com/cloud/learn/apache-kafka)
- [What is Kafka? - YouTube](https://www.youtube.com/watch?v=aj9CDZm0Glc)
- [Stanford CS224N: NLP with Deep Learning | Winter 2021 | Lecture 1 - Intro & Word Vectors - YouTube](https://www.youtube.com/watch?v=rmVRLeJRkl4&list=PLoROMvodv4rOSH4v6133s9LFPRHjEmbmJ)

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

- [A Decade of Learnings from Y Combinator's CEO Michael Seibel - YouTube](https://www.youtube.com/watch?v=0MGNf1BIuxA)

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

-[1000x faster data manipulation: vectorizing with Pandas and Numpy - YouTube](https://www.youtube.com/watch?v=nxWginnBklU)

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

- [Topic Modeling with SVD & NMF (NLP video 2) - YouTube](https://www.youtube.com/watch?v=tG3pUwmGjsc&t=2142s)

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

- [Topic Modeling & SVD revisited (NLP video 3) - YouTube](https://www.youtube.com/watch?v=lRZ4aMaXPBI)

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
- [Unit Testing for Data Scientists - Hanna Torrence - YouTube](https://www.youtube.com/watch?v=Da-FL_1i6ps)
