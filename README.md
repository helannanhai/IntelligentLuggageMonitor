# IntelligentLuggageMonitor
## Project Overview

BagGuardSystem-BGS is a computer-vision-based system designed for identifying luggage and detecting abnormal situations in public areas. The system detects persons and luggage in video streams and performs trajectory-based analysis to identify and indicate potentially unattended baggage.

## Core Functions

- **Object Detection**: Identifies persons and luggage  
- **Cross-Frame Tracking**: Assigns stable track IDs to each detected object  
- **Luggage Association Analysis**: Infers the corresponding person for each piece of luggage based on spatial relationships  
- **State Analysis**: Analyzes luggage behavior across time to determine its state  
- **Visual Output**: Overlays object information and status indicators on video frames  
- **Event Logging**: Records key events

## Docker Usage

The project supports building Docker images and running containers to maintain a consistent execution environment across different systems.  
A container created from the project image includes all required dependencies.  
Input data and output results can be managed through mounted directories.  
The specific method of using Docker may vary depending on the deployment environment and is not restricted to a single workflow.

## Input / Output

### Input  
The project uses real-time video streams as the input source. The input consists of continuous video frames used for person and luggage detection, tracking, and behavioral analysis. The video source may come from cameras, streaming URLs, or other real-time video interfaces, depending on the deployment environment.

### Output  
The project produces the following two types of output:

**Real-Time Output Video Stream**  
The output video stream includes overlay information generated from the analysis, such as person and luggage markers, along with corresponding state indicators.  
The output remains in the form of continuous frames and can be used for front-end display, visualization, or further downstream processing.

**Alert Information**  
When certain conditions are detected—for example, unattended luggage—the system generates alert information.  
These alerts may be consumed, displayed, or recorded by external systems, and the method of utilization is not limited by the project.

## Legal Notice

The project processes general video data and does not involve any form of personal identity recognition.  
Users must ensure that the acquisition and use of input video comply with local laws and privacy requirements.  
When handling footage from public areas, relevant privacy protection regulations must be followed, and the system should not be used for identity recognition or for any purpose that conflicts with privacy requirements.
