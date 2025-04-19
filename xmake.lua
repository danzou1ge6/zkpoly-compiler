add_rules("mode.debug", "mode.release")
add_cuflags("-diag-suppress 550") -- Suppress warnings for unused variables(because field uses asm, and compiler thinks some variable is unused)
includes("memory_pool")
includes("core/src")