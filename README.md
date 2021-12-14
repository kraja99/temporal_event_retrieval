# Temporal Event Retrieval
Unsupervised retrieval of events before and after an anchor event on news corpora using SBERT embeddings.

### Resources
- [Project Overview](https://docs.google.com/document/d/1xA1EH5rD1dT-DMMsot2ouLQyNIE9sxehfIqFszFD_6E/edit?usp=sharing)
- [EventNarratives Data](https://github.com/wenlinyao/EventCommonSenseKnowledge_dissertation/blob/main/event_knowledge_2.0.zip)
- [TemporalBART](https://github.com/jjasonn0717/TemporalBART)

## Clustering
This project includes implmentations for a clustering approach to retrieve sets of events before and after an anchor on both the News Narratives and Gigaword corpora. They can be run after setting up a python virtualenv and installing the packages in requirements.txt. Running the notebook should be straightforward, however I'll include some examples of how to run the script below. Example outputs and further implementation details are available in the Project Overview linked above.

## Querying
The notebook for querying is implemented specifically for the News Narratives corpus, but should be able to be easily adapted for Gigaword or other corpora. I've included the conda environment specifications in the querying folder so a new conda environment can be constructed from that file including all required packages. The overall implementation is outlined in the Project Overview.
