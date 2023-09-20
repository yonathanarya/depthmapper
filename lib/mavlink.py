from pymavlink import mavutil
import time

class sendDepth:
    def __init__(self,config) -> None:
        self.serialpath = config["mavlink"]["serialpath"]
        self.baudrate = int(config["mavlink"]["baudrate"])

        self.master = mavutil.mavlink_connection(self.serialpath, baud=self.baudrate, source_system=255)
        # self.master = mavutil.mavlink_connection("tcp:192.168.31.134:5762")
        self.wait_heartbeat(self.master)
        
        print("connecting mavlink...")
        self.wait_conn()
        # self.arm_test()
        self.min_measurement = 0 # minimum valid measurement that the autopilot should use
        self.max_measurement = 255 # maximum valid measurement that the autopilot should use
        self.sensor_type = mavutil.mavlink.MAV_DISTANCE_SENSOR_UNKNOWN
        self.sensor_id = 1
        self.orientation = mavutil.mavlink.MAV_SENSOR_ROTATION_NONE # downward facing
        self.covariance = 0
        self.tstart = time.time()
        # self.depth(3)
        

    def wait_heartbeat(self, m):
        """
        Wait for a heartbeat so we know the target system IDs
        """
        print("Waiting for APM heartbeat")
        msg = m.recv_match(type="HEARTBEAT", blocking=True)
        print("Heartbeat from APM (system %u component %u)" % (m.target_system, m.target_component))

    def wait_conn(self):
        """
        Sends a ping to stabilish the UDP communication and awaits for a response
        """
        msg = None
        while not msg:
            self.master.mav.ping_send(
                int(time.time() * 1e6), # Unix time in microseconds
                0, # Ping number
                0, # Request ping of all systems
                0 # Request ping of all components
            )
            msg = self.master.recv_match()
            time.sleep(0.5)
    
    def arm_test(self):
        """
        This method only for testing, arming the motor output
        """
        self.master.mav.command_long_send(
        self.master.target_system,
        self.master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0,
        1, 0, 0, 0, 0, 0, 0)
    
    def depth(self, depth):
        """
        This method send mavlink message to flight controller for rangefinder data
        args:
            depth: distance value in cm
        """
        print("range: " + str(depth))
        self.master.mav.distance_sensor_send(
            int((time.time() - self.tstart) * 1000),
            self.min_measurement,
            self.max_measurement,
            depth,
            self.sensor_type,
            self.sensor_id,
            self.orientation,
            self.covariance)
    
    # def depth(self, depth):
    #     while True:
    #         for i in range (5, 1000, 10):
    #             time.sleep(0.5)
    #             print("range: " + str(depth))
    #             self.master.mav.distance_sensor_send(
    #                 int((time.time() - self.tstart) * 1000),
    #                 self.min_measurement,
    #                 self.max_measurement,
    #                 i,
    #                 self.sensor_type,
    #                 self.sensor_id,
    #                 self.orientation,
    #                 self.covariance)
