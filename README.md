# CITS4012 Assignment Specification

This assignment can be done individually or in teams of two or three(max). We strongly encourage healthy collaboration. See the [University of Western Australia Working in Groups Guide](https://www.uwa.edu.au/students/-/media/Project/UWA/UWA/Students/Docs/STUDYSmarter/SS4-Working-in-Groups.pdf). If your team member does not engage or collaborate, please contact the unit coordinator (Dr. Caren Han) with describing the contributions of each collaborator. We strongly recommend to start working early so that you will have ample time to discover stumbling blocks and ask questions.

## Wikipedia Question and Answering
In this assignment, you are to propose and implement a Wikipedia QA (Question Answering) framework using the Sequence model and different NLP features. The QA framework should have the ability to read documents/texts and answer questions about them. The detailed information for the implementation and submission step are specified in the following sections. Note that lecture note and lab exercises would be a good starting point and baseline for the assignment.

Note that we specified which lectures and labs are highly related!


**Table of Contents**
- 0. [Important Dates and Schedule (Very Important!!)](https://github.com/adlnlp/CITS4012_2023/edit/main/README.md#0important-dates-and-schedule)
- 1. 
- 
- 





## <img src="https://em-content.zobj.net/thumbs/120/microsoft/319/calendar_1f4c5.png" width="30" />0.Important Dates and Schedule
The Important date for the Assignment can be summarised as follows:
- Assignment Specification Release: 20 April 2023
- Assignment Group EOI Due: 23 April 2023
- Assignment Group Release: 24 April 2023
- Assignment Group Revision Due: 26 April 2023
- Assignment Submission Due: 20 May 2023
All deadlines are 11:59 PM (AWST).

NOTE: 
**If you want to do individual or already have group members in mind**, Please Submit Group EOI by EOI Due (23 April 2023 11:59PM)
Otherwise, your group members will be selected by our teaching team. 

**After the Group Release date (24 April 2023): YOU MUST CONTACT YOUR GROUP MEMBER BY THE GROUP REVISION DUE (26 April 2023).**
**If you want to change your group to individual after EOI due,**  you can revise it by the Assignment Group Revision Due. However, in this case, you should get your team member's agreement. Please email the Unit Coordinator with your group member's agreement.

**If you do not respond your group member by the Group Revision Due (26 April 2023)**, and your group member requests for working as an individual, you must work the Assignment individually.



## <img src="https://em-content.zobj.net/thumbs/120/samsung/349/card-file-box_1f5c3-fe0f.png" width="30" /> 1. DataSet [Compulsory]
In this assignment, you are asked to use Microsoft Research WikiQA Corpus. The WikiQA corpus includes a set of question and sentence pairs, which is collected and annotated for research on open-domain question answering. The question sources were derived from Bing query logs, and each question is linked to a Wikipedia page that potentially has the answer. More detail on this data can be found in the paper, [WikiQA: A Challenge Dataset for Open-Domain Question Answering](https://aclweb.org/anthology/D15-1237). 

- **Download datasets**:
You will be provided two datasets, including the [training dataset](https://drive.google.com/uc?id=1SXoGbD9WZHwhpqR-cBw7-8_7Ri06nIb6) and the [testing dataset](https://drive.google.com/uc?id=1TwuDSxlcAFDnTRpF-GRvqRXoR_UsJznH). Both datasets contain the following attributes: QuestionID, Question, DocumentID, DocumentTitle, SentenceID, Sentence, and Label (answer sentence, if label=1). If you want to explore or use the full dataset, you can download it via the [Link](https://www.microsoft.com/en-us/download/details.aspx?id=52419). However, the training and testing split should be followed by the one we provided.

- **Data Wrangling**:
You need to first wrangle the dataset. As you can see in the following Figure 1, each row is based on each sentence of the document. You need to construct three different types of data for training the model: Question, Document and Answer. To construct the document data, you should concatenate (with space or full-stop and space) each sentence that has the same DocumentID. For the answer data, use the sentence that has Label as 1.

Note: 1) Some questions may not have answers. (All labels are 0) 2) Some documents may have multiple questions.

<img src="https://github.com/adlnlp/CITS4012_2023/blob/main/Assi_figure1.PNG"><p align="center">**Figure 1. WikiQA: Raw Data - Sample View**</p>




## <img src="https://em-content.zobj.net/thumbs/120/microsoft/319/calendar_1f4c5.png" width="30" />2.QA Model Implementation
You are to propose and implement the open-ended QA framework using word embedding, different types of feature combinations, and deep learning models. The following architecture describes an overview of QA framework architecture. (Click [This Link](https://github.com/adlnlp/CITS4012_2023/blob/main/assignment_overview.png) to view the high-resolution image.)

<img src="https://github.com/adlnlp/CITS4012_2023/blob/main/assignment_overview.png"><p align="center">**Figure 2. Overview of the Architecture of the WikiQA**</p>


