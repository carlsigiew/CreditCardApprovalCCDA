# Credit Card Approval Prediction

**Team Members:**  
          Trish Bandhekar  
          Craig Lewis  
          Ashka Ashani  
          Tejas Patil  
          Yogen Ghodke  
                
## Project Scope and Business Goal:

**Project Scope:**
The project aims to develop a machine learning model for credit card approval prediction, leveraging Amazon Sagemaker and it's ability to test out multiple models and pick the best one. The focus is on assessing the credit risk of applicants based on historical data and personal information. The goal is to enhance the credit scoring process, balancing transparency and predictive power and create an application that credit card approvers can use to quickly sift through applicants.

**Domain:**
The project operates in the financial industry, specifically in credit risk assessment for credit card applications. Key characteristics include the use of historical data, economic sensitivity, and the need for transparent decision-making. Stakeholders in this domain include financial institutions, regulators, and credit applicants.

**Literature Review:**
The literature review encompasses research papers, case studies, articles, and books related to credit scoring in the financial industry. Relevant topics include the application of logistic regression, machine learning algorithms, and the trade-off between transparency and complexity in credit scoring models. At least five current sources are reviewed and summarized to stay informed about advancements and best practices.

https://www.academia.edu/93999609/Credit_Card_Approval_Prediction_using_Classification_Algorithms

This paper explores the use of machine learning models to predict credit card approval based on various factors, addressing credit risk in banks. The project, implemented in Python using Jupyter notebooks, emphasizes dataset preprocessing to meet high expectations for machine learning model accuracy. Through exploratory data analysis, a predictive model is developed, considering parameters in credit card applications. Three algorithms were analyzed, and the Gradient Boosting Classifier achieved the highest accuracy of 90%, surpassing Support Vector Classifier and Adaboost Classifier.

https://www.hindawi.com/journals/ddns/2021/5080472/

This paper explores credit card default prediction for on-loan users using an XGBoost-LSTM model. With the growth of the credit card business, banks have achieved success in customer retention and market expansion. The on-loan phase involves vast data dimensions, posing challenges in risk identification. Leveraging big data analysis and artificial intelligence, the study focuses on mining and analyzing transaction flow data to predict user defaults. Comparative research between XGBoost and LSTM reveals that while XGBoost's accuracy relies on feature extraction expertise, LSTM achieves higher accuracy without it. The XGBoost-LSTM model exhibits strong classification performance in default prediction, offering insights for applying deep learning algorithms in finance.

https://dl.ucsc.cmb.ac.lk/jspui/bitstream/123456789/4593/1/2018%20BA%20026.pdf

This research addresses the application of machine learning (ML) techniques to predict customer eligibility for credit cards, aiming to mitigate potential credit risks for banks. Traditional credit scoring models often lack accuracy, leading to non-performing credit facilities. The project utilizes Artificial Neural Network (ANN) and Support Vector Mechanism (SVM) models, with Nonlinear SVM outperforming ANN and Linear SVM in terms of accuracy, precision, and recall. The study emphasizes the importance of considering country-specific customer behavior and the impact of factors like COVID-19 in real banking datasets. Additionally, the exploration of Nonlinearity in highly imbalanced class problems with SMORTE application is highlighted as a research area.

https://www.ijsce.org/wp-content/uploads/papers/v11i2/B35350111222.pdf

This paper focuses on predicting credit card approval by addressing credit risk in banks through a machine learning model. The objective is to assess the probability of customer default and potential financial implications. The dataset, comprising both mathematical and non-mathematical elements, undergoes preprocessing to ensure the effectiveness of the AI model. Exploratory data analysis is conducted to inform decision-making. The machine learning model, implemented using Python in Jupyter notebook, achieves an 86% accuracy in predicting credit card approval based on various applicant factors. Despite attempts to further enhance performance through grid search and employing random forest and logistic regression models, the accuracy remains at 86%.

https://ieeexplore.ieee.org/document/9763647

The paper explores credit card approval predictions using logistic regression, linear SVM, and Naïve Bayes classifiers in response to the expanding databases of financial institutions. It highlights the improvement in decision-making methods, moving beyond manual judgment to incorporate statistical analysis, enhancing the reliability and efficiency of credit issuance decisions. Acknowledging the growing significance of machine learning algorithms, the paper evaluates multiple regression models and classifiers, specifically Logistic Regression, Linear Support Vector Classification (Linear SVC), and Naïve Bayes Classifier, based on predetermined performance criteria. The goal is to identify an optimal model with the highest prediction accuracy to enhance the credit scoring process.


**Data Source(s):**
The dataset used for this project is one from kaggle (https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction).
This is a highly reviewed data set where the data has been masked to protect the individual's privacy and data. 

**Domain-specific Challenges:**
In the financial domain, challenges include regulatory compliance, ethical considerations, and potential biases in the data. Privacy is a paramount concern, and adherence to regulations such as GDPR and financial data protection laws must be ensured. Handling imbalanced datasets and avoiding discriminatory biases in the credit scoring process are additional challenges. The dataset also needs to include a balanced amount of data in order to cover all the possible cases.

**KPI’s:**
Key Performance Indicators (KPIs) will play a crucial role in evaluating the success of the credit scoring model. Metrics such as accuracy, precision, recall, and possibly ROC-AUC will be considered. The choice of KPIs will be aligned with the goals of the financial industry, emphasizing the importance of correctly identifying creditworthy applicants while minimizing the risk of defaults. The KPIs will set the stage for optimizing the models and ensuring they meet industry standards and expectations.

## Phase 2 Deliverable 

**S3 Data Storage:**
1. Amazon Web Services (AWS) provides a highly scalable and secure solution for object storage known as Amazon S3 (Simple Storage Service). We utilize this service to store our dataset before performing preprocessing and applying transformations.

2. S3 offers significant advantages by enabling the storage and retrieval of large volumes of data at any given time, making it an ideal choice for hosting extensive datasets, including those related to product information.

3. In the realm of S3, data is organized into buckets, each with a unique name within the S3 namespace, ensuring global uniqueness.

4. S3 proves to be versatile for product-focused projects as it accommodates various types of data, ranging from text files to images and metadata.

5. With fine-grained access controls, S3 provides the ability to manage and control access to your information, determining who can interact with the stored data.

6. Versioning and logging are valuable features offered by S3, aiding in the tracking of changes and providing visibility into who has accessed the dataset over time.

7. S3 objects are easily retrievable through unique URLs, simplifying integration with analytical software and machine learning techniques.
   
**Data Exploration:**

Athena - We make use of AWS Glue for ETL Transformations

**Pre-processing:**



**Data Transformations - AWS Glue:**

**AWS Glue ETL Pipeline:**


**AWS Pipeline:**

![image](https://github.com/carlsigiew/CreditCardApprovalCCDA/assets/25591822/0507f755-30f0-47a0-b5ad-162fb652e61e)
