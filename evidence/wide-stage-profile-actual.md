Actual P128 profile318048: model109.907ms; FFN34.610ms; sliding/full
QKV27.356/4.020ms; sliding/fullO9.495/2.636ms; attention9.629/4.819ms.
The source separates heads, attention, KV commit and tile copy from the original
inputnorm/QKV/O/FFN stages. begin_stage_profile/take_stage_profile are unchanged.
Disabled profiling uses Option marks and performs no wide-path clock reads.
This export is based directly on e5b67d2, contains no paired/grouped model phase
factor, and changes no kernel dispatch, arithmetic or cache order.
