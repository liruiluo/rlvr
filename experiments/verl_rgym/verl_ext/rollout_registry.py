def ensure_hf_rollout_registered() -> None:
    from verl.workers.rollout.replica import RolloutReplica, RolloutReplicaRegistry

    registry = getattr(RolloutReplicaRegistry, "_registry")
    if "hf" in registry:
        return

    def load_hf_replica() -> type[RolloutReplica]:
        class HFReplica(RolloutReplica):
            def get_ray_class_with_init_args(self):
                raise NotImplementedError("HFReplica does not support colocated/standalone server mode.")

            async def launch_servers(self):
                assert len(self.workers) == self.world_size, (
                    f"worker number {len(self.workers)} not equal to world size {self.world_size}"
                )
                assert self.world_size == 1, "HF rollout replica currently supports world_size=1 only."
                self.servers = [self.workers[0]]
                setattr(self, "_server_handle", self.workers[0])
                setattr(self, "_server_address", "hf")

        return HFReplica

    RolloutReplicaRegistry.register("hf", load_hf_replica)
