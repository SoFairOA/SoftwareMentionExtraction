import pandas as pd
import os
import re
import logging
from tqdm import tqdm
from datetime import datetime
import asyncio
from asyncio import Semaphore
from langchain_aws import ChatBedrock
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.text_splitter import SpacyTextSplitter
import time
import random
from botocore.exceptions import ClientError
from langchain_openai import ChatOpenAI
import spacy
import json
import torch
from thinc.api import get_current_ops, require_gpu
import warnings
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from torch.nn.functional import cosine_similarity as torch_cosine_similarity
from typing import List, Dict, Any
from datasets import load_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SEMAPHORE_LIMIT = 10  
semaphore = Semaphore(SEMAPHORE_LIMIT)

software_description = """
A software refers to a program, application, or system designed to perform specific tasks or functions on a computer or device. In research papers, software is often mentioned as tools, platforms, libraries, utilities, or frameworks utilized to facilitate data analysis, simulations, visualization, or other research-related activities. For clarity, software is distinct from methods; methods are procedures applied within a study, whereas software serves as the tool or platform enabling these methods. 
Software names frequently include version numbers, specific editions, or configurations, which are critical details and should be captured if explicitly stated in the text. Only explicitly named software should be extracted, avoiding generic terms unless they form part of the official software name as stated in the paper. Software tools can vary widely in purpose, ranging from statistical analysis software (e.g., SPSS) to data visualization platforms (e.g., Tableau) and specialized simulation tools (e.g., ANSYS Fluent).
Examples of explicitly mentioned software might include sentences such as, “We utilized MATLAB for signal processing,” or “Data analysis was conducted using SPSS.” Software references may also occur alongside descriptions of integration, configuration, or utilization in specific stages of the research. In addition, software names might appear in discussions about methodology, where the software’s role in facilitating a particular research approach is described. Software is commonly referenced in areas like machine learning, statistics, scientific computation, bioinformatics, and more.
In some cases, software names may be implicitly referenced by terms such as “tool,” “platform,” or “system.” However, only named instances of software should be captured, excluding general terms that lack explicit names.
"""

url_description = """
A URL (Uniform Resource Locator) is a web address that points to a specific resource or location on the internet. In academic research, URLs are often included to reference online resources, datasets, software repositories, or documentation, providing direct access to supporting materials or additional information. URLs are commonly associated with software downloads, online tools, and resources hosted on websites like GitHub, institutional repositories, or research-focused data repositories.
In the context of software and research, URLs might link to code repositories, technical documentation, or data sources relevant to the research being discussed. Examples of explicitly stated URLs might include sentences such as, “The dataset is available at https://example.com/dataset” or “Software documentation can be accessed at http://software.org/docs.” URLs should be extracted only when they are explicitly mentioned as part of a software reference or a programming language resource.
URLs are particularly relevant when referencing resources that are not included directly within the paper but serve as crucial external sources for data, tools, or documentation related to the research. URLs can be in various formats, including complete links, secure links (HTTPS), or shortened formats. Only explicitly mentioned URLs that directly reference resources pertinent to the study should be extracted, while general or unrelated URLs can be omitted.
"""

programming_language_description = """
A programming language is a structured language that consists of a set of instructions used to produce specific outputs, enabling researchers to develop software, conduct analyses, and execute computational tasks. In academic research, programming languages are often utilized to create custom scripts, automate processes, or handle complex data operations, and they are frequently referenced in studies that involve data processing, modeling, or simulation. Examples of programming languages commonly used in research include Python, R, and MATLAB, but references to any programming language relevant to the research should be extracted.
Programming languages may be explicitly mentioned in statements about tools and methods used for analysis, data cleaning, or algorithm development. For instance, an article might state, “The algorithm was implemented in Python,” or “Data preprocessing was conducted using R.” Programming languages can also appear in discussions related to custom workflows or pipelines where specific languages were chosen to facilitate particular tasks, often due to their compatibility with certain software libraries or frameworks.
References to programming languages can include details about the specific language, version, or packages used, which can be crucial for reproducibility in scientific research. When a programming language is mentioned explicitly within the text, especially in association with a specific research process or tool, it should be extracted to capture the full context of the study’s methodology.
"""

software_sentences = [
    "We used [Software Name] as the core platform for all statistical analysis.",
    "[Software Name] was critical in visualizing the data trends effectively.",
    "The computational models were built and tested using [Software Name].",
    "For data pre-processing, we utilized specialized modules within [Software Name].",
    "The simulations depended on [Software Name] for complex calculations and accuracy.",
    "Data processing and cleaning were automated through [Software Name].",
    "Our study implemented [Software Name] to handle large-scale data transformations.",
    "Statistical computations were conducted using [Software Name] to ensure rigor.",
    "[Software Name] supported both data analysis and visualization phases in this study.",
    "The experimental workflow was streamlined with automation features in [Software Name].",
    "We leveraged [Software Name] to optimize the performance of our predictive model.",
    "[Software Name] enabled the precise configuration of model parameters.",
    "Data curation and storage were managed efficiently using [Software Name].",
    "The integration of data sources was facilitated by [Software Name] tools.",
    "Our analysis was conducted through the extensive libraries provided by [Software Name].",
    "The study’s data visualization relied on the charting capabilities of [Software Name].",
    "We used [Software Name] to run iterative simulations for scenario testing.",
    "All raw data were processed and normalized with [Software Name] modules.",
    "The final analysis outputs were generated using [Software Name].",
    "For reproducibility, we standardized the workflow within [Software Name]."
]

url_sentences = [
    "The raw dataset can be downloaded from [URL].",
    "Complete documentation for [Software Name] is accessible at [URL].",
    "Research findings and supplementary data are available at [URL].",
    "The study's resources are hosted on a public repository at [URL].",
    "For additional software information, visit the official site at [URL].",
    "Detailed project information is accessible via [URL].",
    "The code repository for our data processing scripts is at [URL].",
    "To replicate this analysis, refer to the dataset at [URL].",
    "The experimental protocol and data files are provided at [URL].",
    "For instructions on tool setup, see [URL].",
    "The online tool we used can be accessed directly at [URL].",
    "All supplemental materials and extended data are stored at [URL].",
    "The source code for the custom scripts is available at [URL].",
    "We retrieved external data resources from [URL].",
    "Documentation on data usage rights is accessible at [URL].",
    "The raw and processed data files are hosted at [URL].",
    "For reproducibility, the environment configuration file is stored at [URL].",
    "Our visualizations can be reviewed through the interactive tool at [URL].",
    "The software installation files are downloadable from [URL].",
    "All open-access research materials for this study are available at [URL]."
]

programming_language_sentences = [
    "The statistical models were developed and tested in [Programming Language].",
    "We relied on [Programming Language] for data manipulation and processing.",
    "[Programming Language] scripts enabled efficient handling of large datasets.",
    "The custom analysis functions were written entirely in [Programming Language].",
    "To preprocess and clean the data, we implemented functions in [Programming Language].",
    "Our visualization scripts were built using libraries in [Programming Language].",
    "[Programming Language] was chosen for its powerful data manipulation capabilities.",
    "All data analysis scripts for this study were coded in [Programming Language].",
    "The custom algorithm was implemented using advanced features in [Programming Language].",
    "Automated reporting and summary generation were achieved through [Programming Language].",
    "For machine learning tasks, we relied on the extensive library support in [Programming Language].",
    "The experimental pipeline was set up and executed with [Programming Language] scripts.",
    "Our data transformation steps were coded in [Programming Language] for efficiency.",
    "The data pipeline integrated multiple processing steps coded in [Programming Language].",
    "We used [Programming Language] to batch process and analyze large datasets.",
    "Machine learning models were developed and validated in [Programming Language].",
    "[Programming Language] enabled us to scale the data processing across multiple cores.",
    "The analytical functions and metrics were custom-built in [Programming Language].",
    "Using [Programming Language], we built a modular workflow for the entire study.",
    "Data visualization charts were dynamically generated with [Programming Language]."
]


KEYWORD_PATTERNS = [
    # Core Software Terminology
    r"\bsoftware\b", r"\btool(?:s|kit(?:s)?|chain(?:s)?|suite(?:s)?)\b",
    r"\bapplication(?:s|\sstack|\sserver|\sgateway|\sproxy|\sclient|\sframework)?\b",
    r"\bprogram(?:s|ming|\senvironment|\sinterface|\slibrary?)\b",
    r"\bpackage(?:s|\smanager|\registry|\srepository|\sindex|\slock)?\b",
    r"\blibrar(?:y|ies|\sfunction(?:s)?|\sdependenc(?:y|ies)|\simport(?:s)?)\b",
    r"\bframework(?:s|\scomponent(?:s)?|\smodule(?:s)?|\sintegration|\sconfiguration)?\b",
    r"\bplatform(?:s|\sservice(?:s)?|\sprovider(?:s)?|\sindependent|\sspecific)?\b",
    
    # Architecture & Components
    r"\barchitect(?:ure|ural|ed|ing)\b", r"\bcomponent(?:s|\sbased|\sdriven|\smodel)?\b",
    r"\bmodule(?:s|\sbased|\sdriven|\sloader|\sbundler|\sexports?)?\b",
    r"\bplugin(?:s|\sarchitecture|\sframework|\smanager|\sregistry|\sloader)?\b",
    r"\bextension(?:s|\spoint(?:s)?|\smanager|\sregistry|\sframework)?\b",
    r"\bmiddleware\b", r"\binterceptor(?:s)?\b", r"\bdecorator(?:s)?\b",
    r"\bfacade(?:s)?\b", r"\badapter(?:s)?\b", r"\bproxy(?:ies)?\b",
    r"\bbridge(?:s)?\b", r"\bwrapper(?:s)?\b", r"\bfactory(?:ies)?\b",
    
    # Technical Infrastructure & Protocols
    r"\bAPI(?:s|\sendpoint(?:s)?|\sgateway|\skey(?:s)?|\stoken(?:s)?|\sversion(?:s)?)\b",
    r"\bSDK(?:s|\stools?|\sframework|\slibrary|\scomponent(?:s)?|\smodule(?:s)?)\b",
    r"\bCLI(?:s|\stool(?:s)?|\scommand(?:s)?|\sinterface|\soption(?:s)?)\b",
    r"\bGUI(?:s|\sbuilder|\sframework|\sdesigner|\slibrary|\scomponent(?:s)?)\b",
    r"\bUI/UX(?:\sdesign|\sframework|\scomponent(?:s)?|\slibrary|\spattern(?:s)?)\b",
    r"\bREST(?:ful|\sAPI|\sendpoint(?:s)?|\sservice(?:s)?|\sclient|\sserver)?\b",
    r"\bGraphQL(?:\sAPI|\sschema|\squery|\smutation|\ssubscription|\sresolvers?)?\b",
    r"\bgRPC(?:\sservice(?:s)?|\sclient|\sserver|\sprotocol|\sstream(?:ing)?)\b",
    
    # Cloud & Infrastructure
    r"\bcloud(?:\snative|\scomputing|\splatform|\sservice(?:s)?|\sprovider(?:s)?)\b",
    r"\bserverless(?:\sfunction(?:s)?|\sarchitecture|\splatform|\sframework)?\b",
    r"\bcontainer(?:s|ization|\sorchestration|\sregistry|\simage(?:s)?|\sruntime)?\b",
    r"\bmicroservice(?:s|\sarchitecture|\spattern(?:s)?|\smesh|\sgateway)?\b",
    r"\bKubernetes(?:\scluster|\snode(?:s)?|\spod(?:s)?|\sservice(?:s)?|\sconfig)?\b",
    r"\bDocker(?:\scontainer(?:s)?|\simage(?:s)?|\scompose|\sswarm|\sfile)?\b",
    r"\bterraform(?:\smodule(?:s)?|\sprovider(?:s)?|\sstate|\splan|\sapply)?\b",
    r"\bansible(?:\splaybook(?:s)?|\srole(?:s)?|\stask(?:s)?|\sinventory)?\b",
    
    # Development Tools & Environments
    r"\bIDE(?:s|\sintegration|\splugin(?:s)?|\sextension(?:s)?|\sfeature(?:s)?)\b",
    r"\beditor(?:s|\sconfig|\splugin(?:s)?|\sextension(?:s)?|\stheme(?:s)?)\b",
    r"\bdebugger(?:s|\stool(?:s)?|\sextension(?:s)?|\sconfiguration|\soption(?:s)?)\b",
    r"\bprofiler(?:s|\stool(?:s)?|\sreport(?:s)?|\sanalysis|\smetric(?:s)?)\b",
    r"\blinter(?:s|\srule(?:s)?|\sconfig|\splugin(?:s)?|\sextension(?:s)?)\b",
    r"\bformatter(?:s|\sconfig|\srule(?:s)?|\splugin(?:s)?|\soption(?:s)?)\b",
    r"\bcompiler(?:s|\soption(?:s)?|\sflag(?:s)?|\swarning(?:s)?|\serror(?:s)?)\b",
    r"\binterpreter(?:s|\soption(?:s)?|\sversion(?:s)?|\sruntime|\sengine)?\b",
    
    # Development Practices & Methodologies
    r"\bDevOps(?:\spipeline|\stools?|\spractices?|\sculture|\sprocess(?:es)?)\b",
    r"\bGitOps(?:\sworkflow|\spipeline|\stools?|\spractices?|\sprocess(?:es)?)\b",
    r"\bMLOps(?:\spipeline|\stools?|\splatform|\sframework|\sprocess(?:es)?)\b",
    r"\bDataOps(?:\spipeline|\stools?|\spractices?|\sprocess(?:es)?|\sflow)?\b",
    r"\bAIOps(?:\splatform|\stools?|\smonitor(?:ing)?|\salert(?:ing)?|\sanalytics)?\b",
    r"\bSRE(?:\spractices?|\stools?|\smetrics?|\smonitoring|\sreliability)?\b",
    r"\bagile(?:\smethodology|\sprocess|\spractices?|\ssprints?|\sboard)?\b",
    r"\bscrum(?:\smaster|\steam|\sboard|\ssprint|\smeeting|\sreview)?\b",
    
    # Testing & Quality Assurance
    r"\btest(?:s|ing|\scase(?:s)?|\ssuite(?:s)?|\sframework|\srunner)?\b",
    r"\bunit[\s-]?test(?:s|ing|\sframework|\srunner|\scase(?:s)?|\ssuite)?\b",
    r"\bintegration[\s-]?test(?:s|ing|\sframework|\senvironment|\scase(?:s)?)\b",
    r"\bE2E[\s-]?test(?:s|ing|\sframework|\sscenario(?:s)?|\scase(?:s)?)\b",
    r"\bTDD(?:\sapproach|\smethodology|\spractices?|\scycle|\sprocess)?\b",
    r"\bBDD(?:\sframework|\stools?|\sscenario(?:s)?|\sfeature(?:s)?|\sstep(?:s)?)\b",
    r"\bCI(?:/CD)?(?:\spipeline|\stools?|\sserver|\sprocess|\sworkflow)?\b",
    r"\bquality(?:\sassurance|\scontrol|\smetrics?|\sreport(?:s)?|\sgate(?:s)?)\b",
    
    # Security & Compliance
    r"\bsecur(?:e|ity)(?:\sprotocol(?:s)?|\sframework|\spolicy|\saudit|\scheck)?\b",
    r"\bauth(?:entication|\sorization|\sprotocol|\sprovider|\sservice|\stoken)?\b",
    r"\bOAuth(?:\sflow|\sprovider|\stoken|\sclient|\sscope|\sgrant)?\b",
    r"\bJWT(?:\stoken|\sauth|\sclaim(?:s)?|\ssignature|\sheader|\spayload)?\b",
    r"\bSSO(?:\sservice|\sprovider|\sintegration|\sprotocol|\sclient)?\b",
    r"\bIAM(?:\spolicy|\srole(?:s)?|\suser(?:s)?|\sgroup(?:s)?|\spermission(?:s)?)\b",
    r"\bVPN(?:\sservice|\sclient|\sserver|\sprotocol|\sconnection|\stunneling)?\b",
    r"\bfirewall(?:\srule(?:s)?|\spolicy|\sconfig|\sfilter(?:s)?|\schain(?:s)?)\b",
    
    # Data & Storage
    r"\bdatabase(?:s|\sserver|\scluster|\ssharding|\sreplication|\sbackup)?\b",
    r"\bSQL(?:\squery|\sserver|\sdatabase|\sschema|\stable(?:s)?|\sview(?:s)?)\b",
    r"\bNoSQL(?:\sdatabase|\sdocument|\scollection|\squery|\sindex(?:es)?)\b",
    r"\bcache(?:\slayer|\smemory|\sstore|\skey(?:s)?|\svalue(?:s)?|\shit)?\b",
    r"\bqueue(?:s|\smanager|\sworker(?:s)?|\smessage(?:s)?|\stopic(?:s)?)\b",
    r"\bstream(?:s|ing|\sprocessing|\spipeline|\sanalytics|\sdata)?\b",
    r"\bETL(?:\spipeline|\sprocess|\sjob(?:s)?|\stask(?:s)?|\sworkflow)?\b",
    r"\bdata(?:\slake|\swarehouse|\smart|\spipeline|\sflow|\smodel(?:s)?)\b",
    
    # Machine Learning & AI
    r"\bML(?:\smodel(?:s)?|\spipeline|\sframework|\salgorithm(?:s)?|\straining)?\b",
    r"\bAI(?:\smodel(?:s)?|\sframework|\salgorithm(?:s)?|\sengine|\ssystem)?\b",
    r"\bdeep[\s-]?learning(?:\smodel(?:s)?|\sframework|\snetwork(?:s)?)\b",
    r"\bneural[\s-]?network(?:s|\sarchitecture|\slayer(?:s)?|\smodel(?:s)?)\b",
    r"\btensor(?:s|\sflow|\sboard|\sprocessing|\scomputation|\sgraph)?\b",
    r"\bfeature(?:s|\sengineering|\sextraction|\sselection|\svector(?:s)?)\b",
    r"\bmodel(?:s|\straining|\sevaluation|\sserving|\sprediction|\sinference)?\b",
    r"\bpredictive(?:\sanalysis|\smodel(?:s)?|\sanalytics|\sengine|\ssystem)?\b",
    
    # Monitoring & Observability
    r"\bmonitor(?:ing|\ssystem|\smetric(?:s)?|\salert(?:s)?|\sdashboard)?\b",
    r"\blog(?:s|ging|\saggregation|\sanalysis|\scollection|\smetric(?:s)?)\b",
    r"\btrace(?:s|ing|\scollection|\sanalysis|\svisualization|\sreport(?:s)?)\b",
    r"\bmetric(?:s|\scollection|\sanalysis|\sdashboard|\sreport(?:s)?)\b",
    r"\balert(?:s|ing|\srule(?:s)?|\spolicy|\snotification(?:s)?|\srouting)?\b",
    r"\bdashboard(?:s|\spanel(?:s)?|\swidget(?:s)?|\sview(?:s)?|\sreport(?:s)?)\b",
    r"\btelemetry(?:\sdata|\scollection|\sanalysis|\sreport(?:s)?|\smetric(?:s)?)\b",
    r"\bhealth[\s-]?check(?:s|\sendpoint(?:s)?|\sprobe(?:s)?|\sstatus)?\b",
    
    # Performance & Optimization
    r"\bperforman(?:t|ce)(?:\smetric(?:s)?|\stest(?:s)?|\stuning|\sanalysis)?\b",
    r"\boptimiz(?:e|ed|ation|ing)(?:\stechnique(?:s)?|\smethod(?:s)?|\sstrategy)?\b",
    r"\bcach(?:e|ing)(?:\sstrategy|\slayer|\spolicy|\sinvalidation|\swarming)?\b",
    r"\bscal(?:e|ing|ability)(?:\sout|\sup|\shorizontal|\svertical|\sauto)?\b",
    r"\bload[\s-]?balanc(?:e|ing|er)(?:\salgorithm|\spolicy|\srule(?:s)?)\b",
    r"\bthrottl(?:e|ing)(?:\spolicy|\srate|\slimit(?:s)?|\srule(?:s)?)\b",
    r"\bfailover(?:\smechanism|\sstrategy|\sprocess|\shandling|\srecovery)?\b",
    r"\bresilien(?:t|ce|cy)(?:\spattern(?:s)?|\sstrategy|\sdesign|\stest(?:s)?)\b",
    
    # Version Control & Release Management
    r"\bversion(?:s|ing|\scontrol|\stag(?:s)?|\sbranch(?:es)?|\srelease)?\b",
    r"\bgit(?:hub|\slab|\sflow|\srepository|\sbranch(?:es)?|\scommit(?:s)?)\b",
    r"\breleas(?:e|ing)(?:\sprocess|\spipeline|\snote(?:s)?|\sversion|\stag)?\b",
    r"\bbranch(?:es|ing)(?:\sstrategy|\smodel|\spattern|\sflow|\smerge)?\b",
    r"\bmerge(?:\srequest(?:s)?|\sconflict(?:s)?|\sstrategy|\spolicy)?\b",
    r"\bcommit(?:s|\smessage|\shistory|\slog|\shash|\srevert|\sundo)?\b",
    r"\btag(?:s|ging)(?:\sstrategy|\sversion(?:s)?|\srelease(?:s)?|\spolicy)?\b",
    r"\bchangelog(?:s|\sentry|\sformat|\sgeneration|\sautomation)?\b"
    
    #Common software names
    r"\bTensorFlow\b", r"\bPyTorch\b", r"\bScikit-learn\b", r"\bNumPy\b", r"\bPandas\b",
    r"\bOpenCV\b", r"\bNLTK\b", r"\bSpaCy\b", r"\bKeras\b", r"\bMXNet\b",
    r"\bXGBoost\b", r"\bLightGBM\b", r"\bCatBoost\b", r"\bHadoop\b", r"\bSpark\b",
    r"\bDocker\b", r"\bKubernetes\b", r"\bAnaconda\b", r"\bJupyter\b", r"\bExcel\b",
    r"\bSPSS\b", r"\bMATLAB\b", r"\bOxmetrics\b", r"Imagescope", r"Living Image", r"STATA"

]


predefined_queries = [
    # Integration & Interoperability Patterns
    r"integrates seamlessly with the ecosystem of",
    r"provides native integration support for",
    r"offers built-in compatibility with",
    r"ensures full interoperability between",
    r"interfaces directly through standardized protocols with",
    r"maintains bidirectional communication with",
    r"exchanges data in real-time with",
    r"synchronizes state automatically with",
    r"connects through secure channels to",
    r"bridges functionality between",
    r"extends core capabilities of",
    r"augments existing functionality in",
    r"wraps underlying implementation of",
    r"leverages native features from",
    r"incorporates modules directly from",
    
    # Development & Implementation Context
    r"data were analyzed by using",
    r"calculations were made using",
    r"developed using modern practices from",
    r"implemented following best practices of",
    r"built upon core principles of",
    r"engineered according to specifications from",
    r"architected following patterns from",
    r"designed in compliance with",
    r"constructed using components from",
    r"assembled using building blocks from",
    r"structured according to guidelines from",
    r"modeled after reference implementation of",
    r"patterned on established practices from",
    r"derived from foundational work in",
    r"inspired by architectural patterns of",
    r"adapted from proven approaches in",
    r"evolved from earlier versions of",
    
    # Technical Dependencies & Requirements
    r"requires the following critical dependencies:",
    r"depends fundamentally on components from",
    r"necessitates prior installation of",
    r"mandates runtime presence of",
    r"prerequisites include latest version of",
    r"requires minimum configuration of",
    r"demands specific setup for",
    r"needs compatible versions of",
    r"relies on core functionality from",
    r"operates exclusively with",
    r"functions optimally with",
    r"performs best when paired with",
    r"achieves full capability with",
    r"requires enterprise licensing for",
    r"needs dedicated resources from",
    
    # Operational & Runtime Relationships
    r"runs within containerized environment of",
    r"executes in runtime context of",
    r"operates under supervision of",
    r"functions within constraints of",
    r"performs operations through",
    r"processes workloads using",
    r"handles requests via",
    r"manages resources through",
    r"coordinates activities with",
    r"orchestrates workflows via",
    r"schedules tasks through",
    r"dispatches events to",
    r"routes traffic through",
    r"load balances using",
    r"scales automatically with",
    
    # Configuration & Setup Patterns
    r"configured using standard options from",
    r"setup requires initialization through",
    r"installation managed via",
    r"deployment automated using",
    r"provisioned through infrastructure as code using",
    r"bootstrapped with configuration from",
    r"initialized using templates from",
    r"configured according to best practices from",
    r"customized using parameters from",
    r"tailored through settings in",
    r"optimized using profiles from",
    r"tuned according to benchmarks from",
    r"adjusted based on metrics from",
    r"calibrated using guidelines from",
    r"parametrized following standards from",
    
    # Performance & Optimization Context
    r"optimized for maximum performance using",
    r"accelerated through implementation of",
    r"enhanced with performance features from",
    r"tuned for optimal throughput using",
    r"improved through integration with",
    r"refined using advanced techniques from",
    r"boosted through optimization with",
    r"streamlined using efficiency patterns from",
    r"maximized through careful integration with",
    r"optimized using proven strategies from",
    r"enhanced through careful tuning of",
    r"accelerated via native support for",
    r"improved through deep integration with",
    r"refined using advanced capabilities of",
    r"boosted through strategic use of",
    
    # Security & Compliance Relationships
    r"secured using enterprise-grade features from",
    r"protected through implementation of",
    r"hardened according to standards from",
    r"authenticated using protocols from",
    r"authorized through integration with",
    r"encrypted using algorithms from",
    r"validated against requirements of",
    r"certified compliant with",
    r"audited according to standards of",
    r"monitored for compliance using",
    r"protected by security features of",
    r"safeguarded through implementation of",
    r"secured via integration with",
    r"protected using enterprise solutions from",
    r"hardened following guidelines from",
    
    # Data Management & Processing
    r"processes data streams using",
    r"manages data lifecycle through",
    r"stores information using",
    r"persists state using",
    r"caches frequently accessed data in",
    r"indexes content using",
    r"analyzes data streams with",
    r"transforms data using",
    r"enriches data through",
    r"validates data against",
    r"normalizes data using",
    r"cleanses data through",
    r"aggregates information using",
    r"summarizes data with",
    r"visualizes information through",
    
    # Testing & Quality Assurance
    r"tested comprehensively using",
    r"validated through integration with",
    r"verified using test suites from",
    r"quality assured through",
    r"benchmarked against standards of",
    r"measured for compliance using",
    r"evaluated using criteria from",
    r"assessed through integration with",
    r"analyzed for quality using",
    r"inspected using tools from",
    r"monitored for quality using",
    r"checked against benchmarks from",
    r"validated through automated tests in",
    r"verified using continuous integration with",
    r"tested using frameworks from",
    
    # Monitoring & Observability
    r"monitored in real-time using",
    r"observed through integration with",
    r"tracked using metrics from",
    r"measured using instrumentation from",
    r"profiled using tools from",
    r"analyzed through integration with",
    r"debugged using capabilities of",
    r"traced using features from",
    r"logged using infrastructure from",
    r"alerted through integration with",
    r"visualized using dashboards from",
    r"reported on using tools from",
    r"diagnosed using features of",
    r"troubleshot using capabilities from",
    r"monitored for health using",
    
    # Deployment & Release Management
    r"deployed using automation from",
    r"released through pipeline built with",
    r"distributed using infrastructure from",
    r"published through integration with",
    r"delivered using continuous deployment from",
    r"rolled out using strategies from",
    r"staged using environments from",
    r"promoted through gates defined in",
    r"versioned according to scheme from",
    r"tagged using conventions from",
    r"branched following strategy from",
    r"merged using workflows from",
    r"released using orchestration by",
    r"deployed through automation in",
    r"distributed via channels in",
    
    # Documentation & Knowledge Management
    r"documented extensively in",
    r"described comprehensively through",
    r"explained in detail within",
    r"illustrated with examples in",
    r"demonstrated through tutorials in",
    r"clarified through guides in",
    r"referenced comprehensively in",
    r"detailed specifically in",
    r"outlined thoroughly in",
    r"specified completely in",
    r"articulated clearly in",
    r"elaborated fully in",
    r"cataloged systematically in",
    r"archived comprehensively in",
    r"maintained with documentation in",
    
    # Support & Maintenance
    r"supported officially through",
    r"maintained actively by",
    r"serviced regularly through",
    r"updated periodically via",
    r"patched automatically through",
    r"fixed promptly using",
    r"enhanced regularly with",
    r"upgraded systematically through",
    r"maintained in accordance with",
    r"supported with SLA from",
    r"backed by enterprise support from",
    r"maintained under agreement with",
    r"serviced through contract with",
    r"supported through partnership with",
    r"maintained with assistance from",
    
    # Cloud & Infrastructure
    r"hosted in cloud infrastructure of",
    r"deployed to cloud platform of",
    r"runs on managed services from",
    r"operates in cloud regions of",
    r"scales using cloud capabilities of",
    r"utilizes cloud services from",
    r"leverages cloud features of",
    r"implements cloud patterns from",
    r"follows cloud architecture of",
    r"adopts cloud practices from",
    r"migrated to cloud platform of",
    r"containerized using services from",
    r"orchestrated using platform of",
    r"managed through cloud console of",
    r"automated using cloud tools from",
    
    # Architecture & Design Patterns
    r"follows architectural patterns from",
    r"implements design principles of",
    r"adheres to guidelines from",
    r"conforms to specifications from",
    r"aligns with architecture of",
    r"structured according to patterns in",
    r"designed following principles of",
    r"architected using patterns from",
    r"modeled after architecture of",
    r"patterned on guidelines from",
    r"based on reference architecture of",
    r"derived from patterns in",
    r"structured following blueprint from",
    r"organized according to architecture of",
    r"designed in alignment with",
    
    # Developer Experience & Tooling
    r"developed using toolchain from",
    r"built with developer tools from",
    r"coded using IDE integration with",
    r"implemented using plugins for",
    r"developed with assistance from",
    r"created using frameworks from",
    r"built leveraging tools from",
    r"constructed using utilities from",
    r"assembled using components from",
    r"developed with support of",
    r"implemented through tooling from",
    r"created with integration to",
    r"built using capabilities of",
    r"developed through platform of",
    r"coded with assistance of"
]



chunk_embedding_cache = {}
query_embedding_cache = {}

def get_chunk_embedding(chunk):
    if chunk not in chunk_embedding_cache:
        embedding = embedding_model.encode(chunk, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
        chunk_embedding_cache[chunk] = embedding
    return chunk_embedding_cache[chunk]

def get_query_embedding(query):
    if query not in query_embedding_cache:
        embedding = embedding_model.encode(query, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
        query_embedding_cache[query] = embedding
    return query_embedding_cache[query]


def save_to_cache(data, cache_path):
    with open(cache_path, "w") as cache_file:
        json.dump(data, cache_file)
    
def load_from_cache(cache_path):
    with open(cache_path, "r") as cache_file:
        return json.load(cache_file)

def normalize_keyword(keyword: str) -> str:
    import unicodedata
    keyword = unicodedata.normalize('NFKD', keyword).encode('ASCII', 'ignore').decode('ASCII')
    keyword = keyword.lower()
    keyword = re.sub(r'\s+', ' ', keyword).strip()
    return keyword

def keyword_based_filtering(chunks, compiled_patterns):
    filtered_chunks = []
    for chunk in chunks:
        for pattern in compiled_patterns:
            if pattern.search(chunk):
                filtered_chunks.append(chunk)
                logger.debug(f"Chunk kept: '{chunk}' matched pattern: '{pattern.pattern}'")
                break 
    logger.info(f"{len(filtered_chunks)} chunks kept out of {len(chunks)}.")
    return filtered_chunks

def convert_keywords_to_patterns(keywords: List[str]) -> List[re.Pattern]:
    """
    Converte una lista di keywords in pattern regex sicuri e compilati.
    
    Args:
        keywords (List[str]): Lista di keywords da convertire.
    
    Returns:
        List[re.Pattern]: Lista di pattern regex compilati.
    """
    escaped_keywords = [re.escape(keyword) for keyword in keywords]

    return [re.compile(pattern, re.IGNORECASE) for pattern in escaped_keywords]

def make_hashable(obj):
    if isinstance(obj, list):
        return tuple(make_hashable(e) for e in obj)
    elif isinstance(obj, dict):
        return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
    else:
        return obj

def save_extracted_entities(new_entities, existing_entities):
    existing_entities_set = set()
    for entity in existing_entities:
        software = entity.get('software', '')
        version = entity.get('version', [])
        
        software_hashable = make_hashable(software)
        version_hashable = make_hashable(version)
        
        key = (software_hashable, version_hashable)
        existing_entities_set.add(key)

    unique_entities = []
    for entity in new_entities:
        if "software" in entity and entity["software"]:
            software = entity.get('software', '')
            version = entity.get('version', [])
            
            software_hashable = make_hashable(software)
            version_hashable = make_hashable(version)
            
            key = (software_hashable, version_hashable)
            if key not in existing_entities_set:
                unique_entities.append(entity)
                existing_entities_set.add(key)
            else:
                logger.debug(f"Duplicate entity skipped: {entity}")
        else:
            logger.warning(f"Invalid entity skipped: {entity}")
    return unique_entities



def configure_spacy_gpu():
    warnings.filterwarnings("ignore", category=FutureWarning, module="thinc.shims.pytorch")
    if torch.cuda.is_available():
        require_gpu()
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        nlp = spacy.load("en_core_web_trf")
        nlp.batch_size = 64 if gpu_memory > 8 else 16   
        logger.info(f"Using GPU with {gpu_memory:.2f} GB memory, batch size: {nlp.batch_size}")
    else:
        nlp = spacy.load("en_core_web_trf")
        logger.warning("GPU not available, using CPU model")
    return nlp

def segment_text_with_overlap(text, num_sentences_per_chunk, overlap_sentences):
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    if not sentences:
        return []
    
    chunks, start = [], 0
    while start + num_sentences_per_chunk <= len(sentences):
        chunk = ' '.join(sentences[start:start + num_sentences_per_chunk])
        chunks.append(chunk)
        start += num_sentences_per_chunk - overlap_sentences

    if start < len(sentences):
        chunk = ' '.join(sentences[-num_sentences_per_chunk:])
        if chunk not in chunks:
            chunks.append(chunk)

    return chunks

def process_text_from_parquet(parquet_file, split_name, split_type, window_size, overlap_sentences, batch_processing, cache_dir="cache", limit=None):
    os.makedirs(cache_dir, exist_ok=True)
    
    cache_filename = f"{Path(parquet_file).stem}_{split_type}_{window_size}_{overlap_sentences}"
    if limit is not None:
        cache_filename += f"_limit{limit}"
    cache_filename += "_cache_2.json"
    
    cache_path = os.path.join(cache_dir, cache_filename)
    
    if os.path.exists(cache_path):
        logger.info(f"Loading cached split results from {cache_path}.")
        cached_results = load_from_cache(cache_path)
        return cached_results['chunks_by_document'], cached_results['num_sentences_per_chunk'], cached_results['overlap_sentences']
    
    df = pd.read_parquet(parquet_file)

    if limit is not None:
        df = df.iloc[:limit]

    chunks_by_document = {}
    num_sentences_per_chunk, overlap_sentences = window_size, overlap_sentences
    
    if batch_processing:
        try:
            batch_size = nlp.batch_size
            logger.info(f"Using batch_size from nlp.batch_size: {batch_size}")
        except NameError:
            logger.warning("nlp.batch_size is not defined. Using default batch size 100.")
            batch_size = 100
    else:
        batch_size = len(df)

    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i + batch_size]
        for _, row in tqdm(batch.iterrows(), desc=f"Processing batch {i // batch_size + 1}", total=len(batch)):
            document_id, document_text = row['id'], row['text']
            if split_type == "complete":
                chunks_by_document[document_id] = [document_text]
            else:
                chunks = segment_text_with_overlap(document_text, num_sentences_per_chunk, overlap_sentences)
                chunks_by_document[document_id] = chunks
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    cache_data = {
        "chunks_by_document": chunks_by_document,
        "num_sentences_per_chunk": num_sentences_per_chunk,
        "overlap_sentences": overlap_sentences
    }
    save_to_cache(cache_data, cache_path)
    logger.info(f"Data cached at {cache_path}.")
    return chunks_by_document, num_sentences_per_chunk, overlap_sentences


def process_text_from_hf_parquet(repo_id, config_name, split_name, split_type, window_size, overlap_sentences, batch_processing, cache_dir="cache", limit=None):
    """
    Carica il dataset da Hugging Face utilizzando load_dataset (usando split_name per specificare lo split del dataset),
    converte in DataFrame Pandas, quindi usa split_type per decidere se splittare il testo.
    """
    os.makedirs(cache_dir, exist_ok=True)
    
    cache_filename = f"{config_name}_{split_name}_{window_size}_{overlap_sentences}"
    if limit is not None:
        cache_filename += f"_limit{limit}"
    cache_filename += "_cache.json"
    
    cache_path = os.path.join(cache_dir, cache_filename)

    if os.path.exists(cache_path):
        logger.info(f"Loading cached split results from {cache_path}.")
        cached_results = load_from_cache(cache_path)
        return cached_results['chunks_by_document'], cached_results['num_sentences_per_chunk'], cached_results['overlap_sentences']

    try:
        logger.info(f"Loading dataset from repository: {repo_id}, config: {config_name}, split: {split_name}")
        dataset = load_dataset(repo_id, config_name, split=split_name)
        logger.info("Dataset loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise

    df = dataset.to_pandas()

    if limit is not None:
        df = df.iloc[:limit]

    chunks_by_document = {}
    num_sentences_per_chunk = window_size

    if batch_processing:
        try:
            batch_size = nlp.batch_size
            logger.info(f"Using batch_size from nlp.batch_size: {batch_size}")
        except NameError:
            logger.warning("nlp.batch_size is not defined. Using default batch size 100.")
            batch_size = 100
    else:
        batch_size = len(df)
        logger.info(f"Using batch_size equal to the number of records: {batch_size}")

    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        for _, row in tqdm(batch.iterrows(), desc=f"Processing batch {i // batch_size + 1}", total=len(batch)):
            document_id, document_text = row['id'], row['text']
            if not isinstance(document_id, str) or not isinstance(document_text, str):
                logger.warning(f"Record missing 'id' or 'text' as string: {row}")
                continue
            
            if split_type == "complete":
                chunks_by_document[document_id] = [document_text]
            else:
                chunks = segment_text_with_overlap(document_text, num_sentences_per_chunk, overlap_sentences)
                chunks_by_document[document_id] = chunks

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("CUDA cache cleared.")

    cache_data = {
        "chunks_by_document": chunks_by_document,
        "num_sentences_per_chunk": num_sentences_per_chunk,
        "overlap_sentences": overlap_sentences
    }
    save_to_cache(cache_data, cache_path)
    logger.info(f"Dataset processing completed and cached at {cache_path}.")
    return chunks_by_document, num_sentences_per_chunk, overlap_sentences


def extract_json_from_response(response_text):
    """
    Cleans and extracts the JSON-like structure from a response.
    """
    try:
        json_start = response_text.find('[')
        json_end = response_text.rfind(']') + 1
        if json_start != -1 and json_end != -1:
            json_text = response_text[json_start:json_end]

            json_text = re.sub(r'\((?!.*")([^)]*)\)', '', json_text)

            json_text = re.sub(r'//(?!.*")([^\n]*)', '', json_text)

            json_text = re.sub(r',\s*([\]}])', r'\1', json_text)

            parsed_json = json.loads(json_text)

            sanitized_json = []
            for obj in parsed_json:
                valid_obj = {
                    "software": obj.get("software", ""),
                    "version": obj.get("version", []),
                    "publisher": obj.get("publisher", []),
                    "url": obj.get("url", []),
                    "language": obj.get("language", [])
                }
                sanitized_json.append(valid_obj)

            return sanitized_json

        else:
            logger.warning("No JSON array found in the response.")
            return None
    except json.JSONDecodeError as e:
        logger.error(f"JSON decoding error: {e}. Problematic text: {response_text}")
        return None

async def execute_with_retries(func, *args, max_retries=8, **kwargs):
    retries = 0
    while retries < max_retries:
        try:
            return await func(*args, **kwargs)
        except ClientError as e:
            if e.response['Error']['Code'] == 'ThrottlingException':
                retries += 1
                wait_time = (1 ** retries) + random.uniform(0, 1)
                logger.warning(f"ThrottlingException: Retry {retries}/{max_retries}, waiting {wait_time} seconds.")
                await asyncio.sleep(wait_time)
            else:
                raise e
        except Exception as e:
            retries += 1
            wait_time = (1 ** retries) + random.uniform(0, 1)
            logger.warning(f"Exception encountered: {e}. Retry {retries}/{max_retries}, waiting {wait_time} seconds.")
            await asyncio.sleep(wait_time)
    raise Exception(f"Max retries exceeded for function {func.__name__}")

def semantic_similarity_filter_torch(chunks, queries, threshold=0.3):
    if not chunks or not queries:
        logger.warning("No chunks or queries to process for semantic similarity.")
        return []

    chunk_embeddings = torch.stack([get_chunk_embedding(chunk) for chunk in chunks])
    query_embeddings = torch.stack([get_query_embedding(query) for query in queries])

    similarities = torch_cosine_similarity(chunk_embeddings.unsqueeze(1), query_embeddings.unsqueeze(0), dim=-1)

    max_similarities, _ = similarities.max(dim=1)

    for idx, sim in enumerate(max_similarities):
        logger.info(f"Chunk {idx} similarity: {sim.item()}")

    relevant_indices = (max_similarities >= threshold).nonzero(as_tuple=True)[0].tolist()
    return [chunks[i] for i in relevant_indices]





#all-MiniLM-L12-v2
#all-MiniLM-L6-v2
#all-mpnet-base-v2
#multi-qa-mpnet-base-cos-v1
nlp = configure_spacy_gpu()
embedding_model = SentenceTransformer("all-mpnet-base-v2")