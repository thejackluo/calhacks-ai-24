# Vify.ai

## Overview

Vify.ai is a sophisticated scam detection system designed to identify and
prevent digital scams using cutting-edge AI and ML technologies. This project
leverages various advanced technologies, including LangChain, asynchronous
fine-tuning, contextual analysis, and real-time learning, to provide a robust
and scalable solution for scam detection.

## Inspiration

The inspiration for Vify.ai came from personal experiences with scams that our
team members and their acquaintances faced:

One of our teammates almost fell for a housing scam while searching for
accommodation in San Francisco, encountering fake listings and deceptive agents.
Another teammate's grandparent was nearly duped into wiring $70,000 in a
long-term, subtle manipulation scam that exploited their trust and kindness over
several months. These incidents highlighted the need for a sophisticated,
reliable scam detection system to protect individuals from falling victim to
such fraud.

## What It Does

Vify.ai offers several key features to detect and prevent scams:

- Identify Potential Scams: Utilizes custom-created models to detect scams by
  analyzing text messages and communications.
- LangChain Integration: Integrates multiple models using LangChain for enhanced
  performance and accuracy.
- Real-Time Fine-Tuning: Continuously updates and fine-tunes models to detect
  new types of scams, ensuring the system remains current and effective.
- Psychological Feature Detection: Analyzes messages for psychological markers,
  enhancing detection capabilities.
- Contextual Understanding: Employs a general model to understand the full
  context of the message, providing more nuanced and accurate detections.

## How We Built It

- Backend: Implemented using Python and Flask.
- Scam Classifier, Psychological Model, General Model: Utilized a combination of
  Hugging Face and various APIs, as well as our custom models.
- Database: Stored new scam data in DynamoDB and used AWS for storage and
  scalability.
- Real-Time Learning: Implemented real-time training and inference to adapt to
  new scam tactics quickly.
- Emotion Detection: Utilized Hume.ai for emotion detection in voice, enhancing
  our scam detection capabilities.

## Challenges We Ran Into

- Data Augmentation: Efficiently fine-tuning models with limited data required
  us to learn and implement advanced data augmentation techniques.
- Handling Images: Managing and analyzing image-based data was challenging due
  to complexity and resource requirements.

## Accomplishments That We're Proud Of

- Real-Time Training and Inference: Achieved real-time capabilities for training
  and inference, ensuring our models remain up-to-date.
- Psychological Feature Detection: Developed robust models to detect
  psychological markers within messages.
- Custom Model Development: Successfully created and implemented our own models
  for scam detection.

## What's Next

- Multimodal Expansion: We aim to expand our project to include multimodal data
  such as video, enhancing our detection capabilities.
- Enhanced API Experience: Improve the API endpoint experience for users,
  ensuring ease of integration and use.
- Regulatory Compliance: Keep security at the forefront of development and
  ensure all data processing and storage practices are compliant with global
  privacy regulations such as GDPR and CCPA.
- Frontend Development: Create a user-friendly frontend for demonstrations and
  user interaction.
