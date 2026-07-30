from __future__ import annotations

import os


SITL_ADDRESSES = {
    "udpin://0.0.0.0:14540",
    "udp://:14540",
}


class MavsdkVehicle:
    def __init__(self, system_address: str = "udpin://0.0.0.0:14540") -> None:
        allow_real = os.getenv("ALLOW_REAL_FLIGHT", "false").lower() == "true"
        if system_address not in SITL_ADDRESSES and not allow_real:
            raise RuntimeError("real flight endpoint disabled; use PX4 SITL or complete an external safety review")
        self.system_address = system_address
        self._drone = None

    async def connect(self) -> None:
        from mavsdk import System

        self._drone = System()
        await self._drone.connect(system_address=self.system_address)

    def _require_drone(self):
        if self._drone is None:
            raise RuntimeError("vehicle is not connected")
        return self._drone

    async def takeoff(self, relative_altitude_m: float) -> None:
        drone = self._require_drone()
        await drone.action.set_takeoff_altitude(relative_altitude_m)
        await drone.action.arm()
        await drone.action.takeoff()

    async def goto(self, latitude_deg: float, longitude_deg: float, absolute_altitude_m: float) -> None:
        drone = self._require_drone()
        await drone.action.goto_location(latitude_deg, longitude_deg, absolute_altitude_m, 0.0)

    async def hold(self) -> None:
        await self._require_drone().action.hold()

    async def rtl(self) -> None:
        await self._require_drone().action.return_to_launch()

    async def land(self) -> None:
        await self._require_drone().action.land()
