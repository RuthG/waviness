#Calculate the sinuoisity of daily Z500 data as in Cattiaux et al. 2016
#Inputs daily Z500 file, one file for each year
#Outputs daily timeseries of sinuosity over whole northern hemisphere
#Note that latitudes in ERA-I go from 90 to -90, so I reverse them prior to calculating

library(ncdf4)
library(geosphere)

#list of years
#ylist <- paste0(1979:2018)

#loop throuh list of years
#for (k in 1:length(ylist)) {
#year <- paste0(ylist[k])

#print(year)

#input daily Z500 file
pathinZ <- paste0('/disco/share/rg419/ERA_5/processed/geopotential_1979_2020_daymean_500.nc')

#input file containing average Z500 over 30-70N
#calculated using cdo fldmean -sellonlatbox,0,360,30,70 in.nc out.nc
#(this could be done in this code - not sure why I did it this way)
pathin_iso <- paste0('/disco/share/rg419/ERA_5/processed/geopotential_1979_2020_daymean_500_30_70N.nc')

#output file
pathout <- paste0('/scratch/rg419/derived_data/exps/ERA_5/daily/sinuosity_daily.nc')


inFileZ <- nc_open(pathinZ)
inFile_iso <- nc_open(pathin_iso)

lon <- ncvar_get(inFileZ,varid = "longitude")

lat <- ncvar_get(inFileZ,varid = "latitude")

z500 <- ncvar_get(inFileZ,varid = "z")

z500_iso <- ncvar_get(inFile_iso,varid = "z")

nlat <- length(lat)
#find latitude index at equator
latineq <- nlat %/% 2

#reverse latitudes (ERA-I latitudes are in reverse order)
#functions used require lats to be ascending order
#comment out if not needed
lat <- lat[nlat:1]

#reverse latitude axis in Z500
z500 <- z500[ ,nlat:1, ]


sinNH <- numeric(0)
#loop through each day
for(j in 1:dim(z500)[3] ) {
      #determine contourline at specified isohypse over NH
      con <- contourLines(x = lon, y = lat[latineq:nlat], z = z500[ ,latineq:nlat,j], nlevels = 1, levels = z500_iso[j] )

      per <- 0
      #loop through each contour line and calculate total perimeter
      #There can be >1 line because of cutoff high/lows
      for (i in 1:length(con)){
    
	con1 <- con[i]

    	lonvec <- unlist(con1[[1]][2], use.names=FALSE)

    	latvec <- unlist(con1[[1]][3], use.names=FALSE)
    	
	mat <- matrix(c(lonvec,latvec),nrow=length(lonvec))
    	
	#per1 <- perimeter(mat, a=6378137, f=1/298.257223563)
	per1 <- perimeter(mat, a=6376000, f=0.)
    	
	per <- per+ per1
     }
     sinNH <- c(sinNH,per)
}
#normalize by length perfectly zonal isohypse
sinNH <- sinNH/25751093.0

#copy time dimension from input file
tdim <- inFileZ$dim[['time']]
mv <- 1.e30

var1 <- ncvar_def('sinNH', 'm', list(tdim),mv)



outFile <- nc_create(pathout,var1)
ncvar_put(outFile,var1,sinNH)


nc_close(outFile)
nc_close(inFileZ)
nc_close(inFile_iso)
#}
