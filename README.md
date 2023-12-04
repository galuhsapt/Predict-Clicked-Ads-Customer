
# Predict Clicked Ads Customer Classification by using Machine Learning
by Mochamad Galuh Saputra

## Problem

A company in Indonesia wants to assess the effectiveness of an advertisement they have aired. This is crucial for the company to determine the extent to which the marketed ad has reached its audience and can attract customers to view the advertisement. By analyzing historical advertisement data and uncovering insights and patterns, it can assist the company in determining marketing targets. The focus of this case is to create a machine learning classification model that functions to identify the right target customers.

## Objective
The objectives of this project are to create a machine learning classification model to predict customer behavior regarding clicking on ads. And to enhance the effectiveness and efficiency of campaigns for targeted efforts to increase revenue.

## Exploratory Data Analysis

**![](https://lh7-us.googleusercontent.com/8o4ZVyr7WIZpdjRfq0cV9vamFfoJEzyjAkLWaKOuBQxOynDaFM33dPyYQLUcfqsp03i70kYCqP-OYMltwuC4ZShh2VqyMJPgqABJcHmX1EiKY6er5HjLBR1kaVS4AuIv7UePFZNrzEXogzfNfpOs5G6_xQ=s2048)**

In the visualization above, it is apparent that females tend to click on ads more than males.

**![](https://lh7-us.googleusercontent.com/X0r0CWGk6BkG3Hei4V4jqdnd8cig3AN88ksUHBXbfjP4x4R-rEkE8XUBoONGht92hutuJz0LF41sVErm6Z7f7lvULbsFDZuSbH4EvkpPe375arZPETUHFeikxDpI4blSPfhYznxbTjAdPgLQ6pOnjeC86w=s2048)**
In the visualization beside, it is evident that the regions of Serang and Cimahi are the cities that click on ads most frequently.

**![](https://lh7-us.googleusercontent.com/1wNsiF2q50c3DGSezO7rKeUgJ3Qtv9850QFlTBIsxhPd6vmEV5aPNVjul7aUj6QbO_a3EKf66mgEjBb0gl5MLZjg5hdDf38c5_HSmzbYZNsRYeiF9s3yhcXuxsclIfglYsxklNBW8Dml873W5gHqAaUopQ=s2048)**

The province with the highest ad clicks is South Kalimantan. However, when viewed overall, the difference is not too significant.

**![](https://lh7-us.googleusercontent.com/pSH5uGDsO7UPl0bbRKxvQR7YeRX3lNg-GTtcY-xi7Z4BsJvBnjPRyNt18s0cW2Zs8N6LDxBPf8gRlLVZzsaKCWF5yTH31g6dMJg1ujWJYDFRC-YgUEEK5_EO4BJ94sUIxmsodYHWJqr8yb9gLOe1_LnIFQ=s2048)**

Finance is the category that is most frequently clicked on its ads.

**![](https://lh7-us.googleusercontent.com/B--QG235efiVk5_su0-vZgp3aSi9_cSLl1h37jd6ocQteX8G2QYZXemixEA5BvaehkIAKzqH52qUW_NwcMVA8vO92mIuKUbBuNy1rdjMcVqV2PbFzNfNPy727Co58ffDRlkavIxf5zfJV_E4yQcwtj9TSw=s2048)**

From the visualization, it is apparent that the time spent viewing the site is quite short. This aligns with the limited internet usage for customers clicking on ads. The age group of customers clicking on ads tends to be adults aged 35-48, with varying incomes ranging from 2.8 to 4.

**![](https://lh7-us.googleusercontent.com/eb_p2PsoA62oBPVK-DXJJUzQGVKPEFosSdSl-vIrGj1kQO9nhKX0msWACoRfwarofPE6FxoyEhPqDD74ThL7nwHbODx9oJ7c4M5Qd3K7_jDM97XF1ZGyESpJQGTpPZxOf2Mj33ZtkKg48Y0WaAOhSUX8jQ=s2048)**

Dari grafik heatmap terlihat bahwa kolom yang berkorelasi terhadap Clicked On Ad adalah
-   Daily Internet Usage
-   Daily Time Spent on Site
-   Age
-   Area Income

## Data Cleaning & PreProcessing
**![](https://lh7-us.googleusercontent.com/hrqPCIjoQGl3AtlhGN2JP7_oSfzYzS_JKSd2Irmv1MKGFRg-y2I1w3_yA-rJX6yNrcTp7MvhAGbLEyZr9WtFi2b-_e0hhQJJIsnbcgKfyiFcymgIuqtIVNlzqxHlfE386Mfi1Bm034CLlJyNRd_uWfJmbQ=s2048)![](https://lh7-us.googleusercontent.com/oUDyENPgirGH9LHnQLIk3-CKL41rsD7qBMIGqXrN4R2WSbXvyxFIT0psbS4-xgJy_WvxW_8v27V7kIuA3MNSaHeEnpMOqDNktYbJ-uqYXnqR9afmdnihv2oyfzRi_fdeAW510btaA_7Mw1Uc82iYScYswg=s2048)**
There are null/missing values in the columns Daily Time Spent on Site, Area Income, Daily Internet Usage, and Male. However, there is no duplicated data. The null values will be dropped as their quantity is relatively small.

**![](https://lh7-us.googleusercontent.com/hERn0Qn8YoTu7XoBih0yt5RPVsVC3DgvfvwoByFVmyCY7Ee240omAyZ11VIAgwHqCpPWSz7Tn8TyWTXmRMelSmJMEKRgVjOcJ4mcpxTWKM-jjZsupf3OpHuXi19m8q27NJ7qymRqgmzqKhc23f-3IzpY6g=s2048)**
The data extracted from the timestamp includes year, month, week, and day. Then, the Timestamp column is dropped to reduce the number of columns that will be used

### Label Encoding
**![](https://lh7-us.googleusercontent.com/8582-6fGzNR7T-8QuJ73WDQUXI1V7GFHiO1o2x7EMBCGBVchkTINdM7ULjlx4cSfKdh0O9X_MG61eqO6vBzxi8nZATyhAgZMu6PGbUerZdfy0lDkt0DCVhRM-o8pecKAkUUTgtNo-CV-I_E4V36_aJJpEQ=s2048)![](https://lh7-us.googleusercontent.com/2DJcQfUQSv4VAy8hYN6H94Z4uWIlY8V_Ou8z6Kec2skuYeolex5iZHha6adre26cS4zUIwu9VxWQ981HukjtfuxK58MYxyvK6n_B1AH2Klgg2ThNTjpd2ArP98Y97MSLAPYj6QE3rioYpX235632ikRKAw=s2048)![](https://lh7-us.googleusercontent.com/CoYE-CqZAjtCNR59dvkLmcm9pBr-WcMcnQgAFAkYpcUpVrIvqlzKDpJd_lpP4dKv1q3kAQL_AGmkP-1sOzIUPE7VeZXcJpYB-bAOaK-dQ86z9YEKx1iOJzZosK8f1WI5PkQ31TTuttbaYZojh8FPPt3USQ=s2048)**
The label encoding is applied to the 'Clicked On Ad' column. Similarly, for the 'Male' column, which has been renamed to 'Gender,' label encoding is performed. One-Hot Encoding is applied to the 'Category' column since its unique values are more than 2. However, the 'City' and 'Province' columns will be dropped during the modeling process due to the curse of dimensionality that would result from applying one-hot encoding

### Train-Test Split
**![](https://lh7-us.googleusercontent.com/Vk2WFLaezC0KzmKPPeoUZshK5ddrbooc1DnihmYfgEFLA-tZevDeY0dJMTn7zBXOgkOqMIisJDmbqJzoJw7Avcjwj8MTFmKkD6ZLS3p3fvh_ictyM--rDyqgeVXzV7TMk0zBTOuWy_AuVA8DH9Fp4eaP-Q=s2048)**
The data is split with the target data being 'Clicked on Ad,' using a 30:70 ratio.

## Modeling & Evaluation
### Experiment 1, Without Normalizing
**
before hyperparameter tuning

![](https://lh7-us.googleusercontent.com/QnjIG3tbFiraabzPFuQjTLFkXpeDSngynEqukwCyc3nOFeBD40y0c-7j5f-pFlEIajBGwXmQyA_Gzw9vJwDgrKvgFc8MtddwFK8pHGJcOxkwBMK2noZDuzKMHVA_HxnpsCSJY-FlVS3I3QtIO-decB8VDg=s2048)

After hyperparameter tuning

![](https://lh7-us.googleusercontent.com/g9J2PEtwU9S7meTMCL1mJYMy9dJpmyYULbUzHEo2KEWhmiSIW91O_W5_zdJ3HmuB4KmhldlUE62Wxpd4ZMwyXIbA5leYgCzLbl_VAAijM1HiS_AWfijPhpuWl6v3u-tNYwG0wrGvn_a0SaW49cflshGNlw=s2048)**
After going through the hyperparameter tuning process, the Random Forest model demonstrates superior performance compared to the Decision Tree model. With higher accuracy and a good balance between precision and recall, Random Forest becomes a more reliable choice for practical applications.

### Experiment 2, With Normalizing
Before hyperparameter tuning
**![](https://lh7-us.googleusercontent.com/F33xmhrTMxYjoGjxogV1tObQcZ2se5gEtp6KgcTRTXu53qYSFk67cQfKjjnrp63fjbem36g6cnwZYWAU7EGk-1rq1mNlPfEBVEIWzoxC6ZBvk2QWD1ZPzuHafoxyDXewssUlWb0aKdw2kEb2bORXew0R1w=s2048)**

After hyperparameter tuning
**![](https://lh7-us.googleusercontent.com/vxCvUDrjv_207IUTtJfnuDkuW7dX-YU-RsbOqh03dvdmgCaWshz8AZ6nBHY6qdjHUF6UZTmB0MhkHlfH_iOWxdduRArIff6dD1J3vyCVr70fyCIOuNSqcwRmz9yu81w3h3H8uAaf9I5kdKujKTkEuI9Jsw=s2048)**
It is evident that normalization with Standard Scaler has improved consistency between the model's performance on the training and test sets, resulting in a better improvement in the Random Forest model compared to the Decision Tree. This indicates that normalization has contributed to enhancing the overall performance of the model.

### Confusion Matrix
**![](https://lh7-us.googleusercontent.com/kBHQTElqbRN-UOi7rw7izaN_iuYUT2bWlv_S2OfvxazafWG2SBv2ZgfeBbDy6tiPPWPtjH1xrsTmUfE5tthaiWbuV_wGqmnla08pgdtVQIrnsXU3P7zNRrT9alAxMpQ5fxAtPQxGkgWcAXmzlqkEQEsS1w=s2048)**
From the confusion matrix, it is observed that there are 139 instances of people clicking on the ad and 136 instances of people not clicking on the ad.

**![](https://lh7-us.googleusercontent.com/Kx1r_tEjJ2QvtWmBKxn4wexvjH6LtxltwKxsvaZfy2mvd6KnuodkN0odmDXIWobAM3BZtPW389ZEDYrJd-LSrayDi09ZppxQG5aHcDUYbtE-2tgK8NtcXu0Cv_OZkSxfEzv-znZCqVn1Gnz36PF6p224dQ=s2048)**
Based on feature importance, it is evident that Daily Internet Usage, Daily Time Spent on Site, and Age are the top three features that have the most impact on Clicked On Ad.

## Business Recommendation
- Development of a More Interactive and Responsive Online Platform: 
By focusing on factors such as Daily Internet Usage and Daily Time Spent on Site, businesses can concentrate on developing a more interactive and responsive online platform. This can enhance user engagement and encourage them to spend more time on the website or application, thereby increasing the likelihood of conversion.

- Targeting a Younger Market Segment: 
The age factor (Age) has a significant influence, indicating that businesses need to consider targeting a younger market segment. Optimized marketing strategies and content tailored to their demographic preferences can help attract interest and enhance engagement from this group.

## Business Simulation

### Business Simulation Without Using Machine Learning Models:
- Total Number of Users: 1000 (500 who clicked on the ad and 500 who did not)
- Conversion Rate (CR): (TP + TN) / Total Users = (139 + 136) / 1000 = 0.275
- Marketing Cost: For example, $5000
- Average Revenue per User (ARPU): For example, $100
- Revenue: CR * Total Users * ARPU = 0.275 * 1000 * 100 = $27,500
- Profit: Revenue - Marketing Cost = $27,500 - $5000 = $22,500

### Business Simulation Using a Machine Learning Model:
- Total Number of Users: 1000 (500 who clicked on the ad and 500 who did not)
- Conversion Rate (CR): (TP + TN) / Total Users = (139 + 136) / 1000 = 0.275 (as a baseline)
- Marketing Cost: For example, $5000
- Average Revenue per User (ARPU): Assuming the use of an ML model, ARPU is estimated to increase to $120.
- Revenue: CR * Total Users * ARPU = 0.275 * 1000 * 120 = $33,000
- Profit: Revenue - Marketing Cost = $33,000 - $5000 = $28,000

From the above comparison, it can be seen that the use of ML can increase revenue by up to $5500 or increase up to 20%.
