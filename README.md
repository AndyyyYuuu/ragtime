# ragtime
This is a prototype of intelligent music score library retrieval and question answering using Senso and Anthropic APIs. 

See some of the earlier tests I did in [thoughts.md](thoughts.md) that convinced me to build an agent layer myself instead of only querying Senso's.  

### Method

1. The score is preprocessed into compact text form that allocates 1 line per measure. 
2. Each measure is then analyzed by Claude Haiku 4.5 to identify notable events in the score. 
3. Both fine-grained notes and high-level description files are saved in Senso for retrieval. 
4. During test time, the agent queries from the database to find musical features. The agent is instructed to validate results with the fine-grained score. 



### Limitations & Future Work

- The score used, *The Planets*, is well-known and likely susceptable to benchmark contamination. Rigourous performance testing should not rely on this score. 
- I'm exploring more fine-grained music encoding methods that are readable by LLMs. 
    - XML is extremely verbose (*The Planets* comes down to over 2 million lines). 
    - There's some funky action going on with the CSV parsing in Senso. It seems like my CSV data is being preprocessed into prose somewhere along the pipeline before reaching Senso's chatbot. 
    - My current encoding scheme seems to work fine, but it's definitely lossy. 

