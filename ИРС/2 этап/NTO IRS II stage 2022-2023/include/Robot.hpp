#pragma once
#include <valarray>

class Robot
{
    std::valarray<double>* motorVoltage;

public:

    struct State
    {
        double leftMotorAngle;
        double rightMotorAngle;
        double leftMotorSpeed;
        double rightMotorSpeed;
    };

    Robot() {}
    Robot(std::valarray<double>* var)
    {
        motorVoltage = var;
    }
    ~Robot() {}

    inline void setMotorVoltage(std::valarray<double> & voltage)
    {
        *motorVoltage = voltage;
    }

    State state;
};
