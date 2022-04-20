//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//

#ifndef SRC_RAISIMGYMENV_HPP
#define SRC_RAISIMGYMENV_HPP

#include <vector>
#include <memory>
#include <unordered_map>
#include <Eigen/Core>
#include "raisim/World.hpp"
#include "raisim/RaisimServer.hpp"
#include "Yaml.hpp"
#include "Reward.hpp"

#define __RSG_MAKE_STR(x) #x
#define _RSG_MAKE_STR(x) __RSG_MAKE_STR(x)
#define RSG_MAKE_STR(x) _RSG_MAKE_STR(x)

#define READ_YAML(a, b, c) RSFATAL_IF(!&c, "Node "<<RSG_MAKE_STR(c)<<" doesn't exist") b = c.template As<a>();

namespace raisim {

using Dtype=float;
using EigenRowMajorMat=Eigen::Matrix<Dtype, -1, -1, Eigen::RowMajor>;
using EigenVec=Eigen::Matrix<Dtype, -1, 1>;
using EigenBoolVec=Eigen::Matrix<bool, -1, 1>;

class RaisimGymEnv {

    public:
        explicit RaisimGymEnv (std::string resourceDir, const Yaml::Node& cfg) : resourceDir_(std::move(resourceDir)), cfg_(cfg) { }

        virtual ~RaisimGymEnv() { close(); };

        /////// implement these methods /////////
        virtual void init() = 0;
        virtual void reset() = 0;
        virtual void observe(Eigen::Ref<EigenVec> ob) = 0;
        virtual float step(const Eigen::Ref<EigenVec>& action) = 0;
        virtual bool isTerminalState(float& terminalReward) = 0;
        ////////////////////////////////////////

        /////// optional methods ///////
        virtual void curriculumUpdate() {};
        virtual void close() { if(server_) server_->killServer(); };
        virtual void setSeed(int seed) {};
        ////////////////////////////////

        void setSimulationTimeStep(double dt) { simulation_dt_ = dt; world_->setTimeStep(dt); }
        void setControlTimeStep(double dt) { control_dt_ = dt; }
        int getObDim() { return obDim_; }
        int getActionDim() { return actionDim_; }
        double getControlTimeStep() { return control_dt_; }
        double getSimulationTimeStep() { return simulation_dt_; }
        raisim::World* getWorld() { return world_.get(); }
        void turnOffVisualization() { server_->hibernate(); }
        void turnOnVisualization() { server_->wakeup(); }
        void startRecordingVideo(const std::string& videoName ) { server_->startRecordingVideo(videoName); }
        void stopRecordingVideo() { server_->stopRecordingVideo(); }
        raisim::Reward& getRewards() { return rewards_; }

    protected:
        std::unique_ptr<raisim::World> world_;
        double simulation_dt_ = 0.0025; // 0.001
        double control_dt_ = 0.01;
        std::string resourceDir_;
        Yaml::Node cfg_;
        int obDim_=0, actionDim_=0;
        std::unique_ptr<raisim::RaisimServer> server_;
        raisim::Reward rewards_;
};

}

#endif //SRC_RAISIMGYMENV_HPP
