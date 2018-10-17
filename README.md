# Descubriendo Patrones de Comportamiento Temporal

Analizamos las `TXs - VS` en un distrito de Lima -`sjl` y construimos clusters de comportamiento semana del consumidor, por cada  `MCCG`.


## Details

Creamos marcas temporales del movimiento de las TXs, para ellos hicimos agrupaciones por cada semana.


Cada unidad de comportamiento (`footprint`) se comporta como una matriz de dimensiones: `t` x `d` 

- `t`: 4 turnos durante el dia `0,6,12,18 hrs.`
- `d`: Cada dia de la semana

## References


`Discovering temporal regularities in retail customers’ shopping behavior`, [here][version]).


[unregistered]:http://docs.julialang.org/en/release-0.5/manual/packages/#installing-unregistered-packages
[version]:https://epjdatascience.springeropen.com/articles/10.1140/epjds/s13688-018-0133-0
[gadfly]:http://gadflyjl.org/stable/
