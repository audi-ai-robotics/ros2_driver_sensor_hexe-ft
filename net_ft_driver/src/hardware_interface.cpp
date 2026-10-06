// Copyright (c) 2022, Grzegorz Bartyzel
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//    * Redistributions of source code must retain the above copyright
//      notice, this list of conditions and the following disclaimer.
//
//    * Redistributions in binary form must reproduce the above copyright
//      notice, this list of conditions and the following disclaimer in the
//      documentation and/or other materials provided with the distribution.
//
//    * Neither the name of the {copyright_holder} nor the names of its
//      contributors may be used to endorse or promote products derived from
//      this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "net_ft_driver/hardware_interface.hpp"

#include <memory>
#include <string>
#include <vector>

#include "hardware_interface/sensor_interface.hpp"
#include "hardware_interface/types/hardware_interface_type_values.hpp"
#include "net_ft_driver/interfaces/net_ft_interface.hpp"
#include "rclcpp/rclcpp.hpp"

const auto kLogger = rclcpp::get_logger("NetFtHardwareInerface");

namespace net_ft_driver
{
NetFtHardwareInterface::NetFtHardwareInterface()
{
}

hardware_interface::CallbackReturn NetFtHardwareInterface::on_init(const hardware_interface::HardwareInfo& info)
{
  if (hardware_interface::SensorInterface::on_init(info) != CallbackReturn::SUCCESS) {
    return hardware_interface::CallbackReturn::ERROR;
  }

  ft_sensor_measurements_ = { { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 } };
  lost_packets_ = 0;
  packet_count_ = 0;
  out_of_order_count_ = 0;
  status_ = 0;

  ip_address_ = info_.hardware_parameters["ip_address"];
  sensor_type_ = info_.hardware_parameters["sensor_type"];
  int rdt_rate = std::stoi(info_.hardware_parameters["rdt_sampling_rate"]);
  int internal_filter_rate = std::stoi(info_.hardware_parameters["internal_filter_rate"]);

  driver_ = NetFTInterface::create(sensor_type_, ip_address_);

  if (!driver_->set_sampling_rate(rdt_rate)) {
    RCLCPP_FATAL(kLogger, "Couldn't set RDT sampling rate of the F/T Sensor!");
    return hardware_interface::CallbackReturn::ERROR;
  }

  if (!driver_->set_internal_filter(internal_filter_rate)) {
    RCLCPP_FATAL(kLogger, "Couldn't set internal low pass filter!");
    return hardware_interface::CallbackReturn::ERROR;
  }

  RCLCPP_INFO(kLogger, "Initialize connection with F/T Sensor");
  return hardware_interface::CallbackReturn::SUCCESS;
}

std::vector<hardware_interface::StateInterface> NetFtHardwareInterface::export_state_interfaces()
{
  std::vector<hardware_interface::StateInterface> state_interfaces;

  for (auto& sensor : info_.sensors) {
    for (size_t j = 0; j < sensor.state_interfaces.size(); ++j) {
      state_interfaces.emplace_back(hardware_interface::StateInterface(sensor.name, sensor.state_interfaces[j].name,
                                                                       &ft_sensor_measurements_[j]));
    }
  }

  state_interfaces.emplace_back(hardware_interface::StateInterface("diagnostic", "packet_count", &packet_count_));
  state_interfaces.emplace_back(hardware_interface::StateInterface("diagnostic", "lost_packets", &lost_packets_));
  state_interfaces.emplace_back(
      hardware_interface::StateInterface("diagnostic", "out_of_order_count", &out_of_order_count_));
  state_interfaces.emplace_back(hardware_interface::StateInterface("diagnostic", "status", &status_));
  return state_interfaces;
}

hardware_interface::CallbackReturn
NetFtHardwareInterface::on_activate(const rclcpp_lifecycle::State& /*previous_state*/)
{
  std::string use_hardware_biasing = info_.hardware_parameters["use_hardware_biasing"];
  if (driver_->start_streaming()) {
    if (use_hardware_biasing == "True" || use_hardware_biasing == "true") {
      if (!driver_->set_bias()) {
        RCLCPP_FATAL(kLogger, "Couldn't zero sensor with software bias!");
        return hardware_interface::CallbackReturn::ERROR;
      }
    } else {
      if (!driver_->clear_bias()) {
        RCLCPP_FATAL(kLogger, "Couldn't clear sensor software bias!");
        return hardware_interface::CallbackReturn::ERROR;
      }
    }
    has_data_time_ = false;
    has_restart_time_ = false;
    RCLCPP_INFO(kLogger, "Started data streaming; awaiting the first F/T packet.");
    return hardware_interface::CallbackReturn::SUCCESS;
  }
  RCLCPP_FATAL(kLogger, "The data stream could not be started!");
  return hardware_interface::CallbackReturn::ERROR;
}

hardware_interface::CallbackReturn
NetFtHardwareInterface::on_deactivate(const rclcpp_lifecycle::State& /*previous_state*/)
{
  if (driver_->stop_streaming()) {
    RCLCPP_INFO(kLogger, "Successfully stoped data streaming!");
    return hardware_interface::CallbackReturn::SUCCESS;
  }
  RCLCPP_FATAL(kLogger, "The data stream could not be stopped!");
  return hardware_interface::CallbackReturn::ERROR;
}

hardware_interface::return_type NetFtHardwareInterface::read(const rclcpp::Time& time,
                                                             const rclcpp::Duration& /*period*/)
{
  std::unique_ptr<SensorData> latest;
  // Keep only the newest sample when the sensor runs faster than ros2_control.
  while (auto data = driver_->receive_data()) {
    latest = std::move(data);
  }

  if (latest) {
    ft_sensor_measurements_ = latest->ft_values;
    lost_packets_ = static_cast<double>(latest->lost_packets);
    packet_count_ = static_cast<double>(latest->packet_count);
    out_of_order_count_ = static_cast<double>(latest->out_of_order_count);
    status_ = static_cast<double>(latest->status);
    last_data_time_ = time;
    has_data_time_ = true;
    return hardware_interface::return_type::OK;
  }

  const bool stream_stale = !has_data_time_ || (time - last_data_time_).seconds() > 0.5;
  const bool can_restart = !has_restart_time_ || (time - last_restart_time_).seconds() > 1.0;
  if (stream_stale && can_restart) {
    if (driver_->start_streaming()) {
      RCLCPP_WARN(kLogger, "No F/T packets received; restarted RDT streaming.");
    } else {
      RCLCPP_ERROR(kLogger, "No F/T packets received and RDT streaming restart failed.");
    }
    last_restart_time_ = time;
    has_restart_time_ = true;
  }
  return hardware_interface::return_type::OK;
}
}  // namespace net_ft_driver

#include "pluginlib/class_list_macros.hpp"

PLUGINLIB_EXPORT_CLASS(net_ft_driver::NetFtHardwareInterface, hardware_interface::SensorInterface)
