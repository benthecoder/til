# 66DaysOfData

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

## Day 12: 10 Sep 2022

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
  - ## information expert principle: the way data flows informs the design

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
