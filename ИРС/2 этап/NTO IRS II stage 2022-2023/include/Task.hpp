#pragma once

#include <opencv2/opencv.hpp>
#include "Robot.hpp"
#include <string>

class Simulation;
class GrayRobotC;
class VioletRobotC;

class Task
{
private:

    Simulation* sim;
    GrayRobotC* grayRobot;
    VioletRobotC* violetRobot;

public:
    Task();
    ~Task();

    void start();
    void stop();
    void complete();

    void sendMessage(std::string msg)
    {};
    std::string getTask() {
        return std::string("P1_W P4_N");
    }

    std::vector<Robot*> getRobots() {return robots;}

    cv::Mat getTaskScene();
    cv::Mat getTaskMapWithZones();


    std::vector<Robot*> robots;
};
