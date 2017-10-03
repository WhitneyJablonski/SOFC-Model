#####################################################################################
#	Main driver function for solving the packed bed reactor problem using DAEs
#	and a DAE solver
#####################################################################################

import cantera as ct
import csv
import os, sys
import math
import numpy as numpy
from pylab import *
import matplotlib.pyplot as plt
from sympy import *
import scipy
from scipy.integrate import ode
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from scipy.sparse import csr_matrix
from sundials import *
import pandas as pd
from collections import defaultdict
import nose
import assimulo.problem
import assimulo.solvers

	
def filen ():
	directory = 'c:/Model/PACK/PyPack/Data'
	x = 0
	for files in os.listdir(directory):
		if files.endswith('.inp'):
			x += 1
	return (x)
	
def PBRFile ():
	
	FilePackBedInput = "C:/Model/PACK/PyPack/PackBed.inp"
	IOPackBed = open(FilePackBedInput)
	
	ProbData = {}
	
	XmolIn = []
	ZetaIn = []
	ZReadZeta = []
	
	for line in iter(IOPackBed):
		
		if line.startswith('%'): 		# don't read in comment lines
			continue
		
		if line.lstrip().startswith(' '): 		# don't read in blank lines
			continue
			
		if line.lstrip().startswith('gas'): 		# read in the gas phase object
			gasdummy, FileCanteraGas, NamePhaseGas = line.split()
			ProbData["gas"] = ct.Solution(FileCanteraGas, NamePhaseGas)
			ProbData["Kgas"] = ProbData["gas"].n_species
			continue
			
		if line.lstrip().startswith('surf'): 	# read in the surface phase object
			surfdummy, FileCanteraSurf, NamePhaseSurf, GasPhase = line.split()
			ProbData["surf"] = ct.Interface(FileCanteraSurf, NamePhaseSurf, [ProbData["gas"]])
			ProbData["Ksurf"] = ProbData["surf"].n_species
			break	
			
	XmolIn = np.zeros((ProbData["Kgas"]), dtype='float32')
	ZetaIn = np.zeros((ProbData["Ksurf"]), dtype='float32')
	XmolInt = np.zeros((ProbData["Kgas"]), dtype='float32')
			
	for line in iter(IOPackBed):
	
		if line.lstrip().startswith('NGeo'):		# read in the number of mesh points
			var, value = line.split()
			IReadNGeo = float(value)
			ProbData["NGeo"] = IReadNGeo
			continue
		
		if line.lstrip().startswith('Mesh'):		# read in mesh bias (left, right, center)
			var, value = line.split()
			IReadMesh = float(value)
			ProbData["Mesh"] = IReadMesh
			continue
		
		if line.lstrip().startswith('XLen'):		# read in length of the reaction channel
			var, value = line.split()
			ZReadXLen = float(value)
			ProbData["XLen"] = ZReadXLen
			continue
		
		if line.lstrip().startswith('Orig'):		# read in origin of the reaction channel
			var, value = line.split()
			ZReadOrig = float(value)
			ProbData["Orig"] = ZReadOrig
			continue
		
		if line.lstrip().startswith('Pore'):
			var, value = line.split()
			ZReadPore = float(value)
			ProbData["Pore"] = ZReadPore
			continue
			
		if line.lstrip().startswith('Taut'):
			var, value = line.split()
			ZReadTaut = float(value)
			ProbData["Taut"] = ZReadTaut
			continue
			
		if line.lstrip().startswith('Diam'):
			var, value = line.split()
			ZReadDiam = float(value)
			ProbData["Diam"] = ZReadDiam
			continue
			
		if line.lstrip().startswith('RPor'):
			var, value = line.split()
			ZReadRPor = float(value)
			ProbData["RPor"] = ZReadRPor	
			continue
			
		if line.lstrip().startswith('SCat'):
			var, value = line.split()
			ZReadSCat = float(value)
			ProbData["SCat"] = ZReadSCat	
			continue
			
		if line.lstrip().startswith('Cond'):
			var, value = line.split()
			ZReadCond = float(value)
			ProbData["Cond"] = ZReadCond	
			continue
			
		if line.lstrip().startswith('Dens'):
			var, value = line.split()
			ZReadDens = float(value)
			ProbData["Dens"] = ZReadDens	
			continue
		
		if line.lstrip().startswith('Cvss'):
			var, value = line.split()
			ZReadCvss = float(value)
			ProbData["Cvss"] = ZReadCvss
			continue
			
		if line.lstrip().startswith('VStr'):
			var, value = line.split()
			ZReadVStr = float(value)
			ProbData["VStr"] = ZReadVStr
			continue
			
		if line.lstrip().startswith('AEnv'):
			var, value = line.split()
			ZReadAEnv = float(value)
			ProbData["AEnv"] = ZReadAEnv
			continue
			
		if line.lstrip().startswith('HEnv'):
			var, value = line.split()
			ZReadHEnv = float(value)
			ProbData["HEnv"] = ZReadHEnv
			continue
		
		if line.lstrip().startswith('TEnv'):
			var, value = line.split()
			ZReadTEnv = double(value) + 273.15
			ProbData["TEnv"] = ZReadTEnv
			continue
			
		if line.lstrip().startswith('Pres'):
			var, value = line.split()
			ZReadPres = double(value) * ct.one_atm
			ProbData["Pres"] = ZReadPres
			continue
			
		if line.lstrip().startswith('Temp'):
			var, value = line.split()
			ZReadTemp = double(value) + 273.15
			ProbData["Temp"] = ZReadTemp
			continue
		
		if line.lstrip().startswith('Xmol'):
			var, NameK, value = line.split()
			XMolK = double(value)
			k = ProbData["gas"].species_index(NameK)
			XmolIn[k] = XMolK
			ProbData["Xmol"] = XmolIn
			continue
		
		if line.lstrip().startswith('Zeta'):
			var, NameK, value = line.split()
			XMolK = double(value)
			k = ProbData["surf"].species_index(NameK)
			ZetaIn[k] = XMolK
			ProbData["Zeta"] = ZetaIn
			continue

		if line.lstrip().startswith('PInt'):
			var, value = line.split()
			ZReadPInt = double(value) * ct.one_atm
			ProbData["PInt"] = ZReadPInt
			continue
		
		if line.lstrip().startswith('TInt'):
			var, value = line.split()
			ZReadTInt = double(value) + 273.15
			ProbData["TInt"] = ZReadTInt
			continue
			
		if line.lstrip().startswith('VInt'):
			var, value = line.split()
			ZReadVInt = double(value)
			ProbData["VInt"] = ZReadVInt
			continue	
			
		if line.lstrip().startswith('XInt'):
			var, NameK, value = line.split()
			XMolK = double(value)
			k = ProbData["gas"].species_index(NameK)
			XmolInt[k] = XMolK
			ProbData["XInt"] = XmolIn
			continue
			
		if line.lstrip().startswith('AInt'):
			var, value = line.split()
			ZReadAInt = double(value)
			ProbData["AInt"] = ZReadAInt
			continue	
		
		if line.lstrip().startswith('ATol'):
			var, value = line.split()
			ZReadATol = double(value)
			ProbData["ATol"] = ZReadATol
			continue	
		
		if line.lstrip().startswith('RTol'):
			var, value = line.split()
			ZReadRTol = double(value)
			ProbData["RTol"] = ZReadRTol
			continue
		
		if line.lstrip().startswith('TBEG'):
			var, value = line.split()
			ZReadTBEG = double(value)
			ProbData["TBEG"] = ZReadTBEG
			continue
		
		if line.lstrip().startswith('TEND'):
			var, value = line.split()
			ZReadTEND = double(value)
			ProbData["TEND"] = ZReadTEND
			continue
			
		if line.lstrip().startswith('TDLT'):
			var, value = line.split()
			ZReadTDLT = double(value)
			ProbData["TDLT"] = ZReadTDLT
			continue
			
		if line.lstrip().startswith('Engr'):
			var, value = line.split()
			IsPackEngr = int(value)
			continue
		
		if line.lstrip().startswith('MIEC'):
			var, value = line.split()
			IsPackMIEC = int(value)
			continue
		
		if line.lstrip().startswith('AMem'):
			var, value = line.split()
			ZReadAMem = double(value)
			continue

		if line.lstrip().startswith('End'):
			break
		
	IOPackBed.close()
	
	return (ProbData)

def GeoMeshOne (LocUnit):
	B = 1.10
	BP = B + 1.0
	BM = B - 1.0
	
	Fact = pow((BP / BM),(1.0 - LocUnit))
	PTXStr = (BP - BM * Fact) / (1.0 + Fact)
	
	return (PTXStr)
	
def GeoMeshTwo (LocUnit):
	B = 1.000002E+00
	A = 0.500000E+00
	
	BP1 = B + 1.0E+00
	BM1 = B - 1.0E+00
	
	BPA = B + 2.0E+00 * A
	BMA = B - 2.0E+00 * A
	
	Fact = pow((BP1 / BM1), ((LocUnit - A) / (1.0E+00 - A)))
	PTXStr = (BPA * Fact - BMA) / (2.0 * A + 1.0) / (1.0 + Fact)
	
	return (PTXStr)

def GeoDef (ProbData):
	# set up the geometric mesh and any stretching bias
	XLTGeo = 0
	TLXGeo = ProbData["XLen"]
	NTXGeo = ProbData["NumPackGeo"] - 1
	
	IStretch = ProbData["Mesh"]
	
	if NTXGeo == 1:
		PTXDXD = TLXGeo
		PTXDXC = TLXGeo
		PTXStr = TLXGeo
	elif NTXGeo > 1:
	# Determine stretching from the input file
	# 1) setup the uniform mesh
		DeltaX = 1.0 / (NTXGeo - 1.0)

		XLOCUni = np.zeros((ProbData["NumPackGeo"],))
	
		for i in range (0, int(NTXGeo)):
			XLOCUni[i] = i * DeltaX
	
		PTXStr = np.zeros((int(NTXGeo),))
	# 2) No stretching
		for i in range (1, int(NTXGeo)):
			PTXStr[i] = XLOCUni[i]
	
	# 3) Stretched to the left
		if IStretch == 1:
			for i in range (0, int(NTXGeo)):
				PTXStr[i] = GeoMeshOne(XLOCUni[i])
	
		for i in range (0, int(NTXGeo)):
			PTXStr[i] = PTXStr[i] * ProbData["XLen"]
	
	# calculate the cell distances
		PTXDXD = np.zeros((int(NTXGeo),), dtype='d')
		for i in range (0, int(NTXGeo)-1):
			PTXDXD[i] = PTXStr[i+1] - PTXStr[i]
	
		PTXDXD[NTXGeo-1] = PTXDXD[int(NTXGeo)- 2]
	
	# calculate the cell size
		ProbData["PTXDXC"] = np.zeros((int(NTXGeo),), dtype='d')
		ProbData["PTXDXC"][0] = 0.5 * PTXDXD[0]
		ProbData["PTXDXC"][NTXGeo-1] = 0.5 * PTXDXD[int(NTXGeo) - 2]
		for i in range (1, int(NTXGeo) - 1):
			ProbData["PTXDXC"][i] = 0.5 * (PTXDXD[i] + PTXDXD[i - 1])
	
	return(PTXStr, PTXDXD, ProbData["PTXDXC"])
	
def KozenyPerm (ProbData):
	VolPore = ProbData["Pore"]
	VolSold = 1.0 - ProbData["Pore"]
	
	VolRato = VolPore / VolSold
	
	DiaPore = ProbData["Diam"] * VolRato
	FactEff = ProbData["Pore"] / ProbData["Taut"]
	
	Perm = DiaPore * DiaPore * FactEff / 72.00
	
	return (Perm)

def PBRInitDr (ProbData):
	
	ProbData["gas"].TPX = (ProbData["Temp"], ProbData["Pres"], ProbData["Xmol"])
	
	# Set up numgeo for python array (starting at 0)
	Mesh = ProbData["NGeo"]
	
	# set up solid properties
	ProbData["Tsld"] = ProbData["Temp"]
	
	ProbData["surf"].TP = (ProbData["Tsld"], ProbData["Pres"])
	ProbData["surf"].coverages = ProbData["Zeta"]

	Index = {}
	
	# Generate the index matrix for the simulation
	ProbData["IDPackTsld"] = 0
	ProbData["IDPackTemp"] = ProbData["IDPackTsld"] + 1
	ProbData["IDPackDens"] = ProbData["IDPackTemp"]
	ProbData["IDPackYmas"] = ProbData["IDPackDens"] + 1
	ProbData["IDPackSurf"] = ProbData["IDPackYmas"] + ProbData["Kgas"] - 1
	ProbData["IDPackEdge"] = ProbData["IDPackSurf"] + ProbData["Ksurf"] - 1
	ProbData["NumPackCom"] = ProbData["IDPackEdge"] + 1
	ProbData["NumPackGeo"] = ProbData["NGeo"] + 1
	ProbData["NumPackVar"] = ProbData["NGeo"] * ProbData["NumPackCom"]
	ProbData["IDStirVolm"] = ProbData["NumPackVar"]
	ProbData["IDStirTemp"] = ProbData["IDStirVolm"] + 1
	ProbData["IDStirDens"] = ProbData["IDStirTemp"] + 1
	ProbData["IDStirYmas"] = ProbData["IDStirDens"]
	ProbData["IDStirSurf"] = ProbData["IDStirYmas"] + ProbData["Kgas"] - 1
	ProbData["NumStirVar"] = ProbData["Kgas"] + 2 + ProbData["Ksurf"] - 1
	ProbData["NVPackStir"] = ProbData["IDStirSurf"] + ProbData["Ksurf"]
	
	# Set the geometric mesh and grid
	ProbData["PTXGEOM"],ProbData["DXDGEOM"],ProbData["DXCGEOM"] = GeoDef(ProbData)
	
	Perm = KozenyPerm(ProbData)
	ProbData["Perm"] = Perm
	PBRArray = {}
	
	# initialize arrays and fill with the constant values
	PermPack = np.zeros((Mesh,))
	PermPack.fill(ProbData["Perm"])
	ProbData["PermPack"] = PermPack
	
	PorePack = np.zeros((Mesh,))
	PorePack.fill(ProbData["Pore"])
	ProbData["PorePack"] = PorePack
	
	TautPack = np.zeros((Mesh,))
	TautPack.fill(ProbData["Taut"])
	ProbData["TautPack"] = TautPack
	
	RPorPack = np.zeros((Mesh,))
	RPorPack.fill(ProbData["RPor"])
	ProbData["RPorPack"] = RPorPack
	
	DiamPack = np.zeros((Mesh,))
	DiamPack.fill(ProbData["Diam"])
	ProbData["DiamPack"] = DiamPack

	SCatPack = np.zeros((Mesh,))
	SCatPack.fill(ProbData["SCat"])
	ProbData["SCatPack"] = SCatPack
	
	CondPack = np.zeros((Mesh,))
	CondPack.fill(ProbData["Cond"])
	ProbData["CondPack"] = CondPack
	
	DensPack = np.zeros((Mesh,))
	DensPack.fill(ProbData["Dens"])
	ProbData["DensPack"] = DensPack
	
	CvssPack = np.zeros((Mesh,))
	CvssPack.fill(ProbData["Cvss"])
	
	TEnvPack = np.zeros((Mesh,))
	TEnvPack.fill(ProbData["TEnv"])
	
	AEnvPack = np.zeros((Mesh,))
	AEnvPack.fill(ProbData["AEnv"])
	
	HEnvPack = np.zeros((Mesh,))
	HEnvPack.fill(ProbData["HEnv"])
	
	# initialize arrays for non-constants
	ProbData["PresPack"] = np.zeros((Mesh,), dtype='d')
	ProbData["TempPack"] = np.zeros((Mesh,), dtype='d')
	ProbData["RhoGPack"] = np.zeros((Mesh,), dtype='d')
	ProbData["YmasPack"] = np.zeros((Mesh,ProbData["Kgas"]), dtype='d')
	ProbData["SitePack"] = np.zeros((Mesh,ProbData["Ksurf"]), dtype='d')
	ProbData["ZetaPack"] = np.zeros((Mesh,ProbData["Ksurf"]), dtype='d')
	ProbData["TsldPack"] = np.zeros((Mesh,), dtype='d')
	ProbData["DensPack"] = np.zeros((Mesh,), dtype='d')
	ProbData["CvssPack"] = np.zeros((Mesh,), dtype='d')
	ProbData["TEnvPack"] = np.zeros((Mesh,), dtype='d')
	ProbData["AEnvPack"] = np.zeros((Mesh,), dtype='d')
	ProbData["HEnvPack"] = np.zeros((Mesh,), dtype='d')
	
	# get the gas properties
	Temp, Pres, Xmol = ProbData["gas"].TPX
	Ymas = ProbData["gas"].Y
	RhoG = ProbData["gas"].density

	# get the solid properties
	Tsld = ProbData["surf"].T
	Site = ProbData["surf"].density_mole
	Zeta = ProbData["surf"].X
	
	ProbData["SitePack"].fill(Site)
	
	# set up the arrays for non-constants
	for i in range (0, int(Mesh)):
		ProbData["PresPack"][i] = Pres
		ProbData["TempPack"][i] = Temp
		ProbData["RhoGPack"][i] = RhoG
		ProbData["TsldPack"][i] = Tsld
		for k in range (0,ProbData["Kgas"]):
			ProbData["YmasPack"][i,k] = Ymas[k]
		
		for k in range (0,ProbData["Ksurf"]):
			ProbData["ZetaPack"][i,k] = Zeta[k]
	
	# set up stirred reactor 
	ProbData["VolmStir"] = ProbData["VStr"]
	ProbData["PresStir"] = Pres
	ProbData["TempStir"] = Temp
	ProbData["RhoGStir"] = RhoG
	
	# define stir arrays
	ProbData["YmasStir"] = Ymas
	ProbData["ZetaStir"] = Zeta + 1

	return
	
def PBRRead (ProbData):
	# Read in the previous solution file as an initial guess for solving the
	# complex problem
	
	FileRestart = "C:/Model/PACK/PyPack/PackBedRestart.dat"
	
	d = pd.read_csv(FileRestart, delim_whitespace=True)
	NumPackOld = 15
	ITC = 0
	
	# Set up numgeo for python array (starting at 0)
	Mesh = ProbData["NGeo"] - 1
	
	# set up arrays for read in values
	PTXOLDM = np.zeros((Mesh - 1,))
	VarOLDM = np.zeros((ProbData["NumPackGeo"] - 1,))
	
	#alldata = IORestart.readlines() 				# all data in one list
	
	PTXOLDM = double(d["15"][0:NumPackOld])			# old mesh points
	ITC = ITC + NumPackOld
	ITD = NumPackOld * 2
	VarOLDM = double(d["15"][ITC:ITD])				# old pressure points
	ProbData["PresPack"] = interp1d(PTXOLDM, VarOLDM)(ProbData["PTXGEOM"])	# new interpolated pressure points
	
	ITC = ITC + NumPackOld
	ITD = ITD + NumPackOld
	VarOLDM = double(d["15"][ITC:ITD])				# old temperature points
	ProbData["TempPack"] = interp1d(PTXOLDM, VarOLDM)(ProbData["PTXGEOM"])	# new interpolated temperature points
	
	ITC = ITC + NumPackOld
	ITD = ITD + NumPackOld
	VarOLDM = double(d["15"][ITC:ITD])				# old density points
	ProbData["RhoGPack"] = interp1d(PTXOLDM, VarOLDM)(ProbData["PTXGEOM"])	# new interpolated density points
	
	ITC = ITC + NumPackOld
	ITD = ITD + NumPackOld
	for k in range (0, ProbData["Kgas"]):
		VarOLDM = double(d["15"][ITC:ITD])			# old mass fraction points
		ITC = ITC + NumPackOld
		ITD = ITD + NumPackOld
		ProbData["YmasPack"][:,k] = interp1d(PTXOLDM, VarOLDM)(ProbData["PTXGEOM"])	# new interpolated mass fractions
	
	for k in range (0, ProbData["Ksurf"]):
		VarOLDM = double(d["15"][ITC:ITD])			# old zeta (site fraction) points
		ITC = ITC + NumPackOld
		ITD = ITD + NumPackOld
		ProbData["ZetaPack"][:,k] = interp1d(PTXOLDM, VarOLDM)(ProbData["PTXGEOM"])	# new interpolated zeta (site fracs)
	
	VarOLDM = double(d["15"][ITC:ITD])				# old solid temp points
	ProbData["TsldPack"] = interp1d(PTXOLDM, VarOLDM)(ProbData["PTXGEOM"])	# new interpolated solid temp points
	
	# stirred reactor at the end of packed bed reactor old parameter read in
	ITC = ITC + NumPackOld
	ITD = ITD + NumPackOld
	ProbData["PresStir"] = double(d["15"][ITC])					# stirred reactor pressure
	ITC = ITC + 1
	ProbData["TempStir"] = double(d["15"][ITC])					# stirred reactor temperature
	ITC = ITC + 1
	ProbData["RhoGStir"] = double(d["15"][ITC])					# stirred reactor density
	ITC = ITC + 1
	ProbData["VolmStir"] = double(d["15"][ITC])					# stirred reactor volume
	
	ITC = ITC + 1
	ITD = ITC + ProbData["Kgas"]
	
	ProbData["YmasStir"] = double(d["15"][ITC:ITD])
	
	ITC = ITC + ProbData["Kgas"]
	ITD = ITD + ProbData["Ksurf"]
	
	ProbData["ZetaStir"] = double(d["15"][ITC:ITD])
	
	return()

def PBRNorm (ProbData):
	# normalize the input mass and mole fractions to unity
	
	Ymas = np.zeros((ProbData["Kgas"],))
	Y = np.zeros((ProbData["Kgas"],))
		
	Zeta = np.zeros((ProbData["Ksurf"],))
	Z = np.zeros((ProbData["Ksurf"],))
	NPG = int(ProbData["NumPackGeo"])
	
	for i in range (0, NPG - 1):
		Pres = ProbData["PresPack"][i]
		Temp = ProbData["TempPack"][i]
	
	# normalize the gas phase mass fractions
		for k in range (0, ProbData["Kgas"]):
			Ymas[k] = ProbData["YmasPack"][i,k]
		
		GasNorm = 0.0
		for k in range (0, ProbData["Kgas"]):
			Y[k] = Ymas[k]
			
			if Y[k] < 0.0:
				Y[k] = 0.0
				
			if Y[k] > 1.0:
				Y[k] = 1.0
				
			GasNorm = GasNorm + Y[k]
			
		for k in range (0, ProbData["Kgas"]):
			Y[k] = Y[k] / GasNorm
			
		ProbData["gas"].TPY = (Temp, Pres, Y)
		
		for k in range (0, ProbData["Kgas"]):
			ProbData["YmasPack"][i,k] = Y[k]
		
		ProbData["RhoGPack"][i] = ProbData["gas"].density
		
	# normalize the surface coverage fractions
		Tsld = ProbData["TsldPack"][i]
			
		for k in range (0, ProbData["Ksurf"]):
			Zeta[k] = ProbData["ZetaPack"][i,k]
			
		SurfNorm = 0.0
		for k in range (0, ProbData["Ksurf"]):
			Z[k] = Zeta[k]
			
			if Z[k] < 0.0:
				Z[k] = 0.0
			
			if Z[k] > 1.0:
				Z[k] = 1.0
				
			SurfNorm = SurfNorm + Z[k]
		
		for k in range (0, ProbData["Ksurf"]):
			Z[k] = Z[k] / SurfNorm
		
		ProbData["surf"].TP = (Tsld, Pres)
		ProbData["surf"].coverages = (Z)
		
		for k in range (0, ProbData["Ksurf"]):
			ProbData["ZetaPack"][i,k] = Z[k]
			
	return
	
def PBRVar (ProbData):
	# generate the variable to integrate over
	# generate the zeros variable array for solving the ODE (packed bed reactor)	
	
	PackVar = np.zeros((ProbData["NVPackStir"],), dtype='d')
	Ymas = np.zeros((ProbData["Kgas"],), dtype='d')
	Y = np.zeros((ProbData["Kgas"] - 1,), dtype='d')
	Zeta = np.zeros((ProbData["Ksurf"],), dtype='d')
	Z = np.zeros((ProbData["Ksurf"] - 1,), dtype='d')
	NPG = int(ProbData["NumPackGeo"])
	
	for i in range (0, NPG - 1):
		IDXODE = i * ProbData["NumPackCom"] + ProbData["IDPackTemp"]
		PackVar[IDXODE] = ProbData["TempPack"][i]
		
		IDXODE = i * ProbData["NumPackCom"] + ProbData["IDPackDens"] + 1
		PackVar[IDXODE] = ProbData["RhoGPack"][i]
		
		for k in range (0, ProbData["Kgas"]):
			Ymas[k] = ProbData["YmasPack"][i,k]
		
		for k in range(0, ProbData["Kgas"] - 1):
			Y[k] = Ymas[k]
		
		for k in range (0, ProbData["Kgas"] - 1):
			IDXODE = i * ProbData["NumPackCom"] + ProbData["IDPackYmas"] + k + 1
			PackVar[IDXODE] = Y[k]
		
		for k in range (0, ProbData["Ksurf"]):
			Zeta[k] = ProbData["ZetaPack"][i,k]
			
		for k in range (0, ProbData["Ksurf"]- 1):
			Z[k] = Zeta[k]
			
		for k in range (0, ProbData["Ksurf"] - 1):
			IDXODE = i * ProbData["NumPackCom"] + ProbData["IDPackSurf"] + k + 1
			PackVar[IDXODE] = Z[k]
		
		IDXODE = i * ProbData["NumPackCom"] + ProbData["IDPackTsld"]
		PackVar[IDXODE] = ProbData["TsldPack"][i]
		
	PackVar[ProbData["IDStirVolm"]] = ProbData["VolmStir"]
	PackVar[ProbData["IDStirTemp"]] = ProbData["TempStir"]
	PackVar[ProbData["IDStirDens"]] = ProbData["RhoGStir"]
	
	for k in range(0, ProbData["Kgas"] - 1):
		Y[k] = ProbData["YmasStir"][k]
			
	for k in range (0, ProbData["Kgas"] - 1):
		PackVar[ProbData["IDStirYmas"] + k + 1] = Y[k]
	
	for k in range (0, ProbData["Ksurf"] - 1):
		Z[k] = Zeta[k]
			
	for k in range (0, ProbData["Ksurf"] - 1):
		PackVar[ProbData["IDStirSurf"] + k + 1] = Z[k]
	
	ProbData["GODEVar"] = PackVar
	
	return 

def FluxIn (ProbData):
	
	#################################################################################
	# establish the inlet flux of heat and mass to the system
	#################################################################################
	VeloInlet = ProbData["VInt"]
	PresInlet = ProbData["PInt"]
	TempInlet = ProbData["TInt"]
	XMolInlet = ProbData["XInt"]
	
	ProbData["gas"].TPX = (TempInlet, PresInlet, XMolInlet)
	
	YmasInlet = ProbData["gas"].Y
	RhoGInlet = ProbData["gas"].density
	
	FluxInlet = np.zeros((ProbData["Kgas"],), dtype='d')
	for k in range (0, ProbData["Kgas"]):
		FluxInlet[k] = RhoGInlet * VeloInlet * YmasInlet[k]
	
	ProbData["W"] = ProbData["gas"].molecular_weights
	
	ProbData["H"] = (ProbData["gas"].standard_enthalpies_RT) * ct.gas_constant * TempInlet
	
	HeatInlet = 0.0
	for k in range (0, ProbData["Kgas"]):
		HeatInlet = HeatInlet + FluxInlet[k] * ProbData["H"][k] / ProbData["W"][k]
	
	return(FluxInlet, HeatInlet)

def FluxOut (ProbData):
	#################################################################################
	# establish the outlet flux of heat and mass to the system
	#################################################################################
	IXC = int(ProbData["NumPackGeo"]) - 2
	ITC = IXC
	
	PTXL = ProbData["PTXGEOM"][IXC] - 0.5 * ProbData["DXCGEOM"][IXC]
	PTXR = ProbData["PTXGEOM"][IXC]
	
	PRESL = ProbData["PresPack"][ITC]
	PRESR = ProbData["PresStir"]
	
	TEMPL = ProbData["TempPack"][ITC]
	TEMPR = ProbData["TempStir"]
	
	YMASL = np.zeros((ProbData["Kgas"],), dtype='d')
	YMASR = np.zeros((ProbData["Kgas"],), dtype='d')
	for k in range (0, ProbData["Kgas"]):
		YMASL [k] = ProbData["YmasPack"][ITC,k]
		YMASR [k] = ProbData["YmasStir"][k]
		
	# PBR properties at the cell interfaces
	Perm = ProbData["PermPack"][ITC]
	Pore = ProbData["PorePack"][ITC]
	Taut = ProbData["TautPack"][ITC]
	RPor = ProbData["RPorPack"][ITC]
	Diam = ProbData["DiamPack"][ITC]
	
	DGMHeat, DGMFlux = DustyGas(Pore, Taut, RPor, Diam, Perm, PTXR, PTXL,
								TEMPL, TEMPR, PRESL, PRESR, YMASL, YMASR, ProbData)
	
	for k in range (0, ProbData["Kgas"]):
		DGMFlux[k] = DGMFlux[k] * ProbData["W"][k]
	
	return(DGMFlux, DGMHeat)

def DGM (Por, Tau, RPo, Dia, PTR, PTL, TL, TR, PL, PR, YL, YR):
	# solve the dusty gas model
	
	# set the transport manager to the dusty gas model
	dg = ct.DustyGas(FileCanteraGas)
	dg.porosity = Por
	dg.tortuosity = Tau
	dg.mean_pore_radius = RPo
	dg.mean_particle_diameter = Dia
	
	DGMFlux = np.zeros((Kgas,))
	Dist = PTR - PTL
	
	# setup properties on the left side
	gas.TPY = (TL, PL, YL)
	XMOLL = gas.X
	DENSL = gas.density
	CONCL = gas.density * XMOLL
	ENTHL = gas.standard_enthalpies_RT * ct.gas_constant * TL
		
	# setup properties on the right side
	gas.TPY = (TR, PR, YR)
	XMOLR = gas.X
	DENSR = gas.density
	CONCR = gas.density * XMOLR
	ENTHR = gas.standard_enthalpies_RT * ct.gas_constant * TR
	
	DGMFlux = dg.molar_fluxes(TL, TR, DENSL, DENSR, YL, YR, Dist)
	
	DGMHeat = 0.0
	for k in range (1,Kgas - 1):
		if DGMFlux[k] >= 0:
			DGMHeat = DGMHeat + ENTHL[k] * DGMFlux[k]
		else:
			DGMHeat = DGMHeat + ENTHR[k] * DGMFlux[k]
	
	# convert molar fluxes from DGM to mass fluxes
	for k in range (1,Kgas - 1):
		DGMFlux [k] = W[k] * DGMFlux[k]
		
	return(DGMFlux, DGMHeat)

def DustyGasCoeffs (radiusPore, porosity, tortuosity, ProbData):
	# Calculate the dusty gas coefficients
	# establish the work array for the coefficients
	DiffKnu = np.zeros((ProbData["Kgas"],), dtype='d')
	DiffDGM = np.zeros((ProbData["Kgas"],ProbData["Kgas"]), dtype='d')
	
	T, P, X = ProbData["gas"].TPX
		
	Fact = (2.0E+00/3.0E+00) * radiusPore * np.sqrt((8.0E+00 * ct.gas_constant * T / np.pi))
	
	for k in range(0, ProbData["Kgas"]):
		DiffKnu [k] = Fact / np.sqrt(ProbData["W"][k])
	
	DiffBin = ProbData["gas"].binary_diff_coeffs
	
	Fact = porosity / tortuosity
	
	DiffKnu = DiffKnu * Fact
	DiffBin = DiffBin * Fact
	
	for i in range (0, ProbData["Kgas"]):
		DiffDGM[i,i] = 1.0E+00 / DiffKnu[i]
		
		for j in range (0, ProbData["Kgas"]):
			if i != j:
				DiffDGM[i,j] = - X[i] / DiffBin[i,j]
				DiffDGM[i,i] = DiffDGM[i,i] + X[j] / DiffBin[i,j]
				
	DiffDGM = np.linalg.inv(DiffDGM)
				
	return(DiffDGM, DiffKnu)
	
def DustyGas (Por, Tau, RPo, Dia, Per, PTR, PTL, TL, TR, PL, PR, YL, YR, ProbData):
	# Dusty gas model calculated "by hand" not using Cantera
	
	permeability = Per
	porosity = Por
	tortuosity = Tau
	radiusPore = RPo
	
	DustyFlux = np.zeros((ProbData["Kgas"],), dtype='d')
	FluxPres = np.zeros((ProbData["Kgas"],), dtype='d')
	FluxConc = np.zeros((ProbData["Kgas"],), dtype='d')
	
	Dist = PTR - PTL
	
	# set the gas phase for the left side
	ProbData["gas"].TPY = (TL, PL, YL)
	XL = ProbData["gas"].X
	CL = ProbData["gas"].concentrations
	HL = np.array(ProbData["gas"].standard_enthalpies_RT,dtype='d') * ct.gas_constant * TL
	
	# set the gas phase for the left side
	ProbData["gas"].TPY = (TR, PR, YR)
	XR = ProbData["gas"].X
	CR = ProbData["gas"].density_mole * XR
	HR = np.array(ProbData["gas"].standard_enthalpies_RT,dtype='d') * ct.gas_constant * TR
	
	PU = 0.5 * (PL + PR)
	TU = 0.5 * (TL + TR)
	XU = 0.5 * (XL + XR)
	CU = 0.5 * (CL + CR)
	
	PU = PL
	TU = TL
	XU = XL
	CU = CL
	
	ProbData["gas"].TPX = (TU, PU, XU)
	
	eta = ProbData["gas"].viscosity
	
	DiffDGM, DiffKnu = DustyGasCoeffs(radiusPore, porosity, tortuosity, ProbData)
	
	Conc = CR - CL
	FC = np.dot(DiffDGM, Conc)
	
	FluxConc = - FC/Dist
	
	Velocity = - (permeability / eta) * (PR - PL) / Dist
	
	for k in range (0, ProbData["Kgas"]):
		FluxPres[k] = 0.0E+00
		for l in range (0, ProbData["Kgas"]):
			FluxPres[k] = FluxPres[k] + DiffDGM[k,l] * CU[l] / DiffKnu[l]
			
		FluxPres[k] = FluxPres[k] * Velocity
		
	for k in range (0, ProbData["Kgas"]):
		DustyFlux[k] = FluxConc[k] + FluxPres[k]
	
	DustyHeat = 0.0
	for k in range (0, ProbData["Kgas"]):
		if DustyFlux[k] >= 0.0:
			DustyHeat = DustyHeat + HL[k] * DustyFlux[k]
		else:
			DustyHeat = DustyHeat + HR[k] * DustyFlux[k]

	return(DustyHeat, DustyFlux)

def PBRDiff(FluxInlet, HeatInlet, FluxOutlet, HeatOutlet, ProbData):
	# calculate the diffusion of heat and mass in the system
	
	FuncTrans = np.zeros((ProbData["NumPackVar"],), dtype='d')
	YMASL = np.zeros((ProbData["NumPackGeo"],ProbData["Kgas"]), dtype='d')
	YMASR = np.zeros((ProbData["NumPackGeo"],ProbData["Kgas"]), dtype='d')
	FlxMassGas = np.zeros((ProbData["Kgas"],), dtype='d')
	VelmPack = np.zeros((ProbData["NumPackGeo"] - 1,), dtype='d')
	NPG = int(ProbData["NumPackGeo"]) - 2
	
	for IX in range (0, NPG):
		IXL = IX
		IXR = IX + 1
		
		ITCL = IXL
		ITCR = IXR
		
		PTXL = ProbData["PTXGEOM"][IXL]
		PTXR = ProbData["PTXGEOM"][IXR]
		
		SIZEL = ProbData["DXCGEOM"][IXL]
		SIZER = ProbData["DXCGEOM"][IXR]
		
		PRESL = ProbData["PresPack"][ITCL]
		PRESR = ProbData["PresPack"][ITCR]
		
		TEMPL = ProbData["TempPack"][ITCL]
		TEMPR = ProbData["TempPack"][ITCR]
		
		for k in range (0, ProbData["Kgas"]):
			YMASL = ProbData["YmasPack"][ITCL]
			YMASR = ProbData["YmasPack"][ITCR]
			
		# establish DGM input parameters at the cell interface
		Perm = 0.5 * (ProbData["PermPack"][ITCL] + ProbData["PermPack"][ITCR])
		Pore = 0.5 * (ProbData["PorePack"][ITCL] + ProbData["PorePack"][ITCR])
		Taut = 0.5 * (ProbData["TautPack"][ITCL] + ProbData["TautPack"][ITCR])
		RPor = 0.5 * (ProbData["RPorPack"][ITCL] + ProbData["RPorPack"][ITCR])
		Diam = 0.5 * (ProbData["DiamPack"][ITCL] + ProbData["DiamPack"][ITCR])
		
		DGMHeat, DGMFlux = DustyGas(Pore, Taut, RPor, Diam, Perm, PTXR, PTXL,
								TEMPL, TEMPR, PRESL, PRESR, YMASL, YMASR, ProbData)
		
		Dist = 0.5 * (SIZEL + SIZER)
		
		#evaluate the left side heat flux
		ProbData["gas"].TPY = (TEMPL, PRESL, YMASL)
		CONDL = ProbData["gas"].thermal_conductivity
		
		# evaluate the right side heat flux
		ProbData["gas"].TPY = (TEMPR, PRESR, YMASR)
		CONDR = ProbData["gas"].thermal_conductivity
		
		if CONDL * CONDR == 0:
			CONDM = 0.0
		else:
			CONDM = (SIZEL + SIZER) / (SIZEL / CONDL + SIZER / CONDR)
		
		FluxHeat = np.abs(-CONDM * (TEMPR - TEMPL) / Dist)
		FluxEngr = DGMHeat + FluxHeat
		
		FlxMassTot = 0
		for k in range (0, ProbData["Kgas"]):
			FlxMassGas[k] = ProbData["W"][k] * DGMFlux [k]
			FlxMassTot = FlxMassTot + FlxMassGas[k]
		
		VelmPack[ITCL] = FlxMassTot / ProbData["PorePack"][ITCL] / ProbData["RhoGPack"][ITCL]
		
		ITCRL = ITCL * ProbData["NumPackCom"] + ProbData["IDPackDens"] + 1
		ITCRR = ITCR * ProbData["NumPackCom"] + ProbData["IDPackDens"] + 1
		
		FuncTrans[ITCRL] = FuncTrans[ITCRL] - FlxMassTot / ProbData["DXCGEOM"][IXL]
		FuncTrans[ITCRR] = FuncTrans[ITCRR] + FlxMassTot / ProbData["DXCGEOM"][IXR]
			
		for k in range (0, ProbData["Kgas"] - 1):
			ITCKL = ITCL * ProbData["NumPackCom"] + ProbData["IDPackYmas"] + k + 1
			ITCKR = ITCR * ProbData["NumPackCom"] + ProbData["IDPackYmas"] + k + 1
			FuncTrans[ITCKL] = FuncTrans[ITCKL] - FlxMassGas[k] / ProbData["DXCGEOM"][IXL]
			FuncTrans[ITCKR] = FuncTrans[ITCKR] + FlxMassGas[k] / ProbData["DXCGEOM"][IXR]
		
		ITCTL = ITCL * ProbData["NumPackCom"] + ProbData["IDPackTemp"]
		ITCTR = ITCR * ProbData["NumPackCom"] + ProbData["IDPackTemp"]
		
		FuncTrans[ITCTL] = FuncTrans[ITCTL] - FluxEngr / ProbData["DXCGEOM"][IXL]
		FuncTrans[ITCTR] = FuncTrans[ITCTL] + FluxEngr / ProbData["DXCGEOM"][IXR]
	
	# conditions at the inlet boundary
	
	IXC = 0
	ITC = IXC
		
	FluxMassTot = 0.0
	for k in range (0, ProbData["Kgas"]):
		FluxMassTot = FluxMassTot + FluxInlet[k]
	
	ITCR = ITC * ProbData["NumPackCom"] + ProbData["IDPackDens"] + 1
	FuncTrans[ITCR] = FuncTrans[ITCR] + FluxMassTot / ProbData["DXCGEOM"][IXC]
	
	for k in range (0, ProbData["Kgas"] - 1):
		ITCK = ITC * ProbData["NumPackCom"] + ProbData["IDPackYmas"] + k + 1
		FuncTrans[ITCK] = FuncTrans[ITCK] + FluxInlet[k] / ProbData["DXCGEOM"][IXC]
	
	ITCT = ITC * ProbData["NumPackCom"] + ProbData["IDPackTemp"]
	FuncTrans[ITCT] = FuncTrans[ITCT] + HeatInlet / ProbData["DXCGEOM"][IXC]
	
	# conditions at the outlet boundary
		
	IXC = NPG
	ITC = IXC
		
	FluxMassTot = 0.0
	for k in range (0, ProbData["Kgas"]):
		FluxMassTot = FluxMassTot + FluxOutlet[k]
	
	ITCR = ITC * ProbData["NumPackCom"] + ProbData["IDPackDens"] + 1
	FuncTrans[ITCR] = FuncTrans[ITCR] - FluxMassTot / ProbData["DXCGEOM"][IXC]
	
	VelmPack[ITC] = FluxMassTot / ProbData["PorePack"][ITC] / ProbData["RhoGPack"][ITC]
	
	for k in range(0, ProbData["Kgas"] - 1):
		ITCK = ITC * ProbData["NumPackCom"] + ProbData["IDPackYmas"] + k + 1
		FuncTrans[ITCK] = FuncTrans[ITCK] - FluxOutlet[k] / ProbData["DXCGEOM"][IXC]
	
	ITCT = ITC * ProbData["NumPackCom"] + ProbData["IDPackTemp"]
	FuncTrans[ITCT] = FuncTrans[ITCT] - HeatOutlet / ProbData["DXCGEOM"][IXC]	
	
	return(FuncTrans)

def PBRReact(ProbData):
	
	FuncReact = np.zeros((ProbData["NumPackVar"],), dtype='d')
	Ymas = np.zeros((ProbData["Kgas"],), dtype='d')
	Y = np.zeros((ProbData["Kgas"],), dtype='d')
	RateGas = np.zeros((ProbData["Kgas"],), dtype='d')
	wdot = np.zeros((ProbData["Kgas"],), dtype='d')
	Zeta = np.zeros((ProbData["Ksurf"],), dtype='d')
	RateSurf = np.zeros((ProbData["Ksurf"],), dtype='d')
	sdot = np.zeros((ProbData["Ksurf"],), dtype='d')
	FlxMassGas = np.zeros((ProbData["Kgas"],), dtype='d')
	FlxCovSurf = np.zeros((ProbData["Ksurf"],), dtype='d')
	NPG = int(ProbData["NumPackGeo"])
	
	for IXC in range (0, NPG - 1):
		ITC = IXC
		
		Pres = ProbData["PresPack"][ITC]
		Temp = ProbData["TempPack"][ITC]
		RhoG = ProbData["RhoGPack"][ITC]
		
		for k in range (0, ProbData["Kgas"]):
			Ymas [k] = ProbData["YmasPack"][ITC,k]
		
		for k in range (0, ProbData["Kgas"]):
			Y[k] = Ymas[k]
		
		ProbData["gas"].TPY = (Temp, Pres, Y)
		
		Tsld = ProbData["TsldPack"][ITC]
		
		for k in range (0, ProbData["Ksurf"]):
			Zeta [k] = ProbData["ZetaPack"][ITC,k]
		
		ProbData["surf"].TP = (Tsld, Pres)
		ProbData["surf"].coverages = (Zeta)
		
		Pore = ProbData["PorePack"][ITC]
		SCat = ProbData["SCatPack"][ITC]
		
		nrxngas = ProbData["gas"].n_reactions
		if nrxngas > 0:
			wdot = ProbData["gas"].net_production_rates
		else:
			for k in range (0, ProbData["Kgas"]): wdot[k] = 0
		
		for k in range (0, ProbData["Kgas"]):
			Index = k
			RateGas[Index] = RateGas[Index] + wdot[Index] * Pore
		
		nrxnsurf = ProbData["surf"].n_reactions
		if nrxnsurf > 0:
			sdot = ProbData["surf"].net_production_rates
		else:
			for k in range (0, ProbData["Kgas"] + ProbData["Ksurf"]): sdot[k] = 0
		
		for k in range (0, ProbData["Kgas"]):
			RateGas[k] = sdot[k] * SCat
			
		for k in range (0, ProbData["Ksurf"]):
			RateSurf[k] = sdot[ProbData["Kgas"] + k]
			
		FlxMassTot = 0
		for k in range (0, ProbData["Kgas"]):
			FlxMassGas[k] = RateGas[k] * ProbData["W"][k]
			FlxMassTot = FlxMassTot + FlxMassGas[k]
		
		# solve for the density
		ITCR = ITC * ProbData["NumPackCom"] + ProbData["IDPackDens"] + 1
		
		FuncReact[ITCR] = FlxMassTot								
		
		# solve for the new mole fractions
		for k in range (0, ProbData["Kgas"]):
			ITCK = ITC * ProbData["NumPackCom"] + ProbData["IDPackYmas"] + k + 1
			FuncReact[ITCK] = FlxMassGas[k]
		
		SiteDensity = ProbData["surf"].density_mole
		
		FlxTotSurf = RateSurf[ProbData["Ksurf"] - 1]
		
		for k in range (0, ProbData["Ksurf"] - 1):
			FlxCovSurf[k] = RateSurf[k] / SiteDensity
			FlxTotSurf = FlxTotSurf + RateSurf[k]
		
		for k in range (0, ProbData["Ksurf"] - 1):
			ITCK = ITC * ProbData["NumPackCom"] + ProbData["IDPackSurf"] + k + 1
			FuncReact[ITCK] = FlxCovSurf[k]
	
	return(FuncReact)
	
def StirDiff(FluxInlet, HeatInlet, ProbData):
	
	global YmasStir, TempStir, PresStir, RhoGStir, ZetaStir
	
	# solve for the stirred tank reactor
	
	ProbData["gas"].TPY = (ProbData["TempStir"], ProbData["PresStir"], ProbData["YmasStir"])
	
	WBarStir = ProbData["gas"].mean_molecular_weight
	
	nspec = ProbData["Kgas"] + ProbData["Ksurf"]
	FluxReact = np.zeros((nspec,), dtype='float32')
	StirFun = np.zeros((ProbData["NVPackStir"] + 1,), dtype='float32')
	HMassGas = np.zeros((ProbData["Kgas"]), dtype='float32')
	
	for k in range (0, Kgas - 1):
		IDS = ProbData["IDStirYmas"] + k + 1
		StirFun [IDS] = 0
		
	FlxMassInt = 0.0
	FlxMassAct = 0.0
	for k in range (0, ProbData["Kgas"]):
		FlxMassInt = FlxMassInt + FluxInlet[k]
		FlxMassAct = FlxMassAct + FluxReact[k]
	
	for k in range (0, ProbData["Kgas"] - 1):
		Rate = FluxInlet[k] - FlxMassInt * YmasStir[k]
		IDS = ProbData["IDStirYmas"] + k + 1
		StirFun[IDS] = StirFun[IDS] + Rate / RhoGStir / VolmStir
		
	for k in range (0, Ksurf - 1):
		IDS = ProbData["IDStirSurf"] + k + 2
		StirFun[IDS] = 0
		
	StirFun[ProbData["IDStirVolm"]] = 0
	
	HBMasStir = ProbData["gas"].enthalpy_mass
	CPMasStir = ProbData["gas"].cp_mass
	
	HMoleGas = ProbData["gas"].standard_enthalpies_RT * ct.gas_constant * TempStir
	for k in range (0, ProbData["Kgas"]):
		HMassGas [k] = HMoleGas[k] / W[k]
			
	HeatInj = HeatInlet - FlxMassInt * HBMasStir
	
	AEnvStir = 0
	HEnvStir = 0
	TEnvStir = 3.00E+02
	
	HeatEnv = AEnvStir * HEnvStir * (TempStir - TEnvStir)
	
	HeatMas = 0
	for k in range (0, ProbData["Kgas"] - 1):
		IDS = ProbData["IDStirYmas"] + k + 1
		HeatMas = HeatMas + StirFun[IDS] * (HMassGas[k] - HMassGas[ProbData["Kgas"] - 1])
	HeatMas = RhoGStir * VolmStir * HeatMas
	
	HeatCap = RhoGStir * VolmStir * CPMasStir
	
	StirFun[ProbData["IDStirTemp"]] = (HeatInj - HeatEnv - HeatMas) / HeatCap
	
	DlogTemp = StirFun[ProbData["IDStirTemp"]] / TempStir
	
	DlogWBar = 0.0
	for k in range (0, ProbData["Kgas"]):
		Rate = (1.0 / W[k] - 1.0 / W[ProbData["Kgas"] - 1]) * StirFun[ProbData["IDStirYmas"] + k + 1]
		DlogWBar = DlogWBar + Rate
	
	DlogWBar = DlogWBar * WBarStir
	
	StirFun[ProbData["IDStirDens"]] = -RhoGStir * (DlogTemp + DlogWBar)
	
	DflowRateDt = StirFun[ProbData["IDStirDens"]] * VolmStir - StirFun[ProbData["IDStirVolm"]] * RhoGStir
	
	FluxMassOut = FlxMassInt + FlxMassAct - DflowRateDt
	
	return(StirFun)
	
def YmasDownload(Tsolv, Rsolv, Ysolv, ProbData):
	# solve for the last y species which is the majority species
	
	Temp = Tsolv
	RhoG = Rsolv
	Ymas = np.zeros((ProbData["Kgas"],), dtype='float32')
	
	YmasLast = 1.0
	num = 0.0
	for k in range (0, ProbData["Kgas"] - 1):
		Ymas[k] = Ysolv[num]
		YmasLast = YmasLast - Ymas[k]
		num = num + 1
		
	Ymas[ProbData["Kgas"] - 1] = YmasLast
	
	for k in range (0, ProbData["Kgas"]):
		Ymas[k] = Ymas[k]
	
	return(Ymas)

def ZetaDownload(Zsolv, ProbData):	
	
	num = 0
	Zeta = np.zeros((ProbData["Ksurf"],), dtype='d')
	
	ZetaLast = 1.0
	for k in range (0, ProbData["Ksurf"] - 1):
		Zeta[k] = Zsolv[num]
		ZetaLast = ZetaLast - Zeta[k]
		num = num + 1
	
	Zeta[ProbData["Ksurf"] - 1] = ZetaLast
	
	return(Zeta)
	
def PBRDownloadVar(GODEUpl, ProbData):	
	# setup the solution array
	
	Ksolv = ProbData["Kgas"] - 1
	Ysolv = np.zeros((Ksolv,))
	Ssolv = ProbData["Ksurf"] - 1
	Zsolv = np.zeros((Ssolv,))
	NPG = int(ProbData["NumPackGeo"])
	
	for i in range (0,NPG-1):
		IDXODE = i * ProbData["NumPackCom"] + ProbData["IDPackTemp"]
		ProbData["TempPack"][i] = GODEUpl[IDXODE]
		
		IDXODE = i * ProbData["NumPackCom"] + ProbData["IDPackDens"] + 1
		ProbData["RhoGPack"][i] = GODEUpl[IDXODE]
		
		Tsolv = ProbData["TempPack"][i]
		Rsolv = ProbData["RhoGPack"][i]
		if Rsolv < 0:
			Rsolv = ProbData["RhoGPack"][i-1]
			
		Pres = ProbData["PresPack"][i]
		
		for k in range (0, Ksolv):
			IDXODE = i * ProbData["NumPackCom"] + ProbData["IDPackYmas"] + k + 1
			Ysolv[k] = GODEUpl[IDXODE]
			
		Ymas = YmasDownload(Tsolv, Rsolv, Ysolv, ProbData)
		
		ProbData["gas"].TDY = (Tsolv, Rsolv, Ymas)
		Pres = ProbData["gas"].P
		
		ProbData["PresPack"][i] = Pres
		
		for k in range (0, ProbData["Kgas"]):
			IDXODE = i * ProbData["NumPackCom"] + ProbData["IDPackYmas"] + k + 1
			ProbData["YmasPack"][i,k] = Ymas[k]
			
		for k in range (0, Ssolv):
			IDXODE = i * ProbData["NumPackCom"] + ProbData["IDPackSurf"] + k + 1
			Zsolv[k] = GODEUpl[IDXODE]
			
		Zeta = ZetaDownload(Zsolv, ProbData)
		
		for k in range (0, ProbData["Ksurf"]):
			IDXODE = i * ProbData["NumPackCom"] + ProbData["IDPackSurf"] + k + 1
			ProbData["ZetaPack"][i,k] = Zeta[k]
		
		IDXODE = i * ProbData["NumPackCom"] + ProbData["IDPackTsld"]
		ProbData["TsldPack"][i] = GODEUpl[IDXODE]
		
	ProbData["VolmStir"] = GODEUpl[ProbData["IDStirVolm"]]
	ProbData["TempStir"] = GODEUpl[ProbData["IDStirTemp"]]
	ProbData["RhoGStir"] = GODEUpl[ProbData["IDStirDens"]]
		
	Tsolv = ProbData["TempStir"]
	Rsolv = ProbData["RhoGStir"]
	
	for k in range(0, Ksolv):
		Ysolv[k] = GODEUpl[ProbData["IDStirYmas"] + k + 1]
		
	Ymas = YmasDownload(Tsolv, Rsolv, Ysolv, ProbData)
	
	ProbData["gas"].TDY = (Tsolv, Rsolv, Ymas)
	
	ProbData["YmasStir"] = Ymas
	Pres = ProbData["gas"].P
	ProbData["PresStir"] = Pres
	
	for k in range (0, Ssolv):
		Zsolv[k] = GODEUpl[ProbData["IDStirSurf"] + k + 1]
	
	return

def FuncC (FluxInlet, HeatInlet, FluxOutlet, HeatOutlet, ProbData):	
	FuncCVar = np.zeros((ProbData["NumPackVar"],), dtype='d')
	
	FuncTrans = PBRDiff(FluxInlet, HeatInlet, FluxOutlet, HeatOutlet, ProbData)
	
	FuncReact = PBRReact(ProbData)
	
	NPV = int(ProbData["NumPackVar"])
	NPG = int(ProbData["NumPackGeo"])
	
	for i in range (0, NPV):
		FuncCVar[i] = FuncTrans[i] + FuncReact[i]
	
	FuncPVar = np.zeros((ProbData["NumPackVar"],), dtype='d')

	for i in range (0, NPV):
		FuncPVar[i] = FuncCVar[i]
	
	for ITC in range (0, NPG - 1):
		ITCR = ITC * ProbData["NumPackCom"] + ProbData["IDPackDens"] + 1
			
		for k in range (0, ProbData["Kgas"] - 1):
			ITCK = ITC * ProbData["NumPackCom"] + ProbData["IDPackYmas"] + k + 1
			FuncPVar[ITCK] = (FuncCVar[ITCK] - ProbData["YmasPack"][ITC,k] * FuncCVar[ITCR]) / ProbData["RhoGPack"][ITC] / ProbData["PorePack"][ITC]
			
		FuncPVar[ITCR] = FuncCVar[ITCR] / ProbData["PorePack"][ITC]
		
	#if IsPackEngr == 2:
	#	FuncPVar = 0
	#else:
	for ITC in range (0, NPG - 1):
		ITCTemp = ITC * ProbData["NumPackCom"] + ProbData["IDPackTemp"]
		ITCTsld = ITC * ProbData["NumPackCom"] + ProbData["IDPackTsld"]
		
		FuncPVar[ITCTemp] = 0.0
		FuncPVar[ITCTsld] = 0.0
	
	return(FuncPVar)
	
def PBRFun (Time, GODEVar,ProbData):
	# define the function to be integrated
	
	# set up the array of the function to be integrated
	GODEFun = np.zeros((ProbData["NVPackStir"],), dtype='d')

	# download variables from solution array
	PBRDownloadVar(GODEVar, ProbData)	
	
	# get the initial flux into the system
	FluxInlet, HeatInlet = FluxIn(ProbData)
	
	# get the flux out of the system (through stirred reactor)
	FluxOutlet, HeatOutlet = FluxOut(ProbData)
	
	PackFun = FuncC(FluxInlet, HeatInlet, FluxOutlet, HeatOutlet, ProbData)
	
	#StirFun = StirDiff(FluxOutlet, HeatOutlet, ProbData)
	
	NPV = int(ProbData["NumPackVar"])
	for i in range (0, NPV):
		GODEFun[i] = PackFun[i]
	
  ### Remove the energy equations
	#for i in range (NumPackVar + 1, NVPackStir):
	#	GODEFun[i] = StirFun[i]
	
	
	#solve the energy equations
	# if IsPackEngr == 1:
		# # HeatReact = 
	
	# # for IXC in range (0,NumPackGeo):
		# # ITC = IXC
	
	# # Pres = PresPack[ITC]
	# # Temp = TempPack[ITC]
	# RhoG = RhoGPack[ITC]
	
	# # for k in range (0, Kgas - 1):
	# # Ymas [k] = YmasPack[ITC,k]
		
		# # gas.TPY = (Pres, Temp, Ymas)
	
	# # Pore = PorePack[ITC]
	# # SCat = SCatPack[ITC]
	
	return (GODEFun)
	
def PBRWrite():
	# write to the restart read file to update the latest solution
	FileWriteRestart = "C:/Model/PACK/PyPack/PackBedRestart%d.dat" % (nfiles)
	IORestart = open(FileWriteRestart, 'w')
	
	IORestart.write("%d\n" %NumPackGeo - 1)
	for ITC in range (0, NumPackGeo - 1):
		IORestart.write("%e\n" %PTXGEOM[ITC])
	
	for ITC in range (0, NumPackGeo - 1):
		IORestart.write("%e\n" %PresPack[ITC])
	
	for ITC in range (0, NumPackGeo - 1):
		IORestart.write("%e\n" %TempPack[ITC])
	
	for ITC in range (0, NumPackGeo - 1):
		IORestart.write("%e\n" %RhoGPack[ITC])
	
	for k in range (0, Kgas):
		for ITC in range (0, NumPackGeo - 1):
			IORestart.write("%e\n" %YmasPack[ITC,k])
		
	for k in range (0, Ksurf):
		for ITC in range (0, NumPackGeo - 1):
			IORestart.write("%e\n" %ZetaPack[ITC,k])
		
	for ITC in range (0, NumPackGeo - 1):
		IORestart.write("%e\n" %TsldPack[ITC])
	
	IORestart.write("%e\n" %PresStir)
	IORestart.write("%e\n" %TempStir)
	IORestart.write("%e\n" %RhoGStir)
	IORestart.write("%e\n" %VolmStir)

	for k in range (0, Kgas):
		IORestart.write("%e\n" %YmasStir[k])
	
	for k in range (0, Ksurf):
		IORestart.write("%e\n" %ZetaStir[k])
	
	return

def PBRSoln(ProbData):
	# write the solutions for the gas and surface phases to files
	fgas = "C:/Model/PACK/PyPack/PBRSolnGas.dat"
	fsurf = "C:/Model/PACK/PyPack/PBRSolnSurf.dat" 
	
	IOGas = open(fgas, 'w')
	IOSurf = open(fsurf, 'w')
	
	glist = []
	for k in range (0,ProbData["Kgas"]):
		name = ProbData["gas"].species_name(k)
		glist.append(name)
		gaslist = ",".join(glist)
	
	Titlegas = ['Position', 'Pres', 'Temp', 'RhoG']
	gastitle = ",".join(Titlegas)
	IOGas.write("{0},{1}\n".format(gastitle, gaslist))
	
	slist = []
	for k in range (0,ProbData["Ksurf"]):
		name = ProbData["surf"].species_name(k)
		slist.append(name)
		surflist = ",".join(slist)
	
	Titlesurf = ['Position', 'Tsld']
	surftitle = ",".join(Titlesurf)
	IOSurf.write("{0},{1}\n".format(surftitle, surflist))
	
	Meshlist = ProbData["PTXDXC"].tolist()
	
	Xlist = []
	Ylist = []
	NPG = int(ProbData["NumPackGeo"])
	Ymas = np.zeros((ProbData["Kgas"],))
	for ITC in range (0, NPG - 1):
		Pres = ProbData["PresPack"][ITC]
		Temp = ProbData["TempPack"][ITC]
		
		for k in range (0, ProbData["Kgas"]):
			Ymas[k] = ProbData["YmasPack"][ITC,k]
		
		ProbData["gas"].TPY = (Temp, Pres, Ymas)
		Xmol = ProbData["gas"].X
		Xlist = Xmol.tolist()
		Ylist = Ymas.tolist()
	
		IOGas.write("{0},{1},{2},{3},{4}\n".format(Meshlist[ITC],ProbData["PresPack"][ITC],ProbData["TempPack"][ITC],ProbData["RhoGPack"][ITC],Ylist))
	
	return

def jaceqn(ProbData):
	
	# set up the Jacobian matrix to initialize the stiff equation solver
	
	Jac = np.zeros((ProbData["NVPackStir"],ProbData["NVPackStir"]))
	
	NumPackCom = int(ProbData["NumPackCom"])
	NumPackGeo = int(ProbData["NumPackGeo"]) - 1
	NumPackVar = int(ProbData["NumPackVar"])
	NumStirVar = int(ProbData["NumStirVar"])
	NVPackStir = int(ProbData["NVPackStir"])
	
	for ITC in range (0, NumPackGeo - 1):
		ITL = ITC - 1
		ITR = ITC + 1
	
		for k in range (0, NumPackCom):
			IRow = (ITC - 1) * NumPackCom + k
		
		for l in range (0, NumPackCom):
			ICol = max( (ITL - 1) * NumPackCom + l, 1 )
			Jac [IRow, ICol] = 1
			
		for l in range (0, NumPackCom):
			ICol = (ITC - 1) * NumPackCom + l
			Jac [IRow, ICol] = 1
		if ITC == NumPackGeo:
			for l in range (0, NumStirVar):
				ICol = min( (ITR - 1) * NumPackCom + l, NVPackStir )
				Jac [IRow, ICol] = 1
		else:
			for l in range (0, NumPackCom):
				ICol = min( (ITR - 1) * NumPackCom + l, NVPackStir )
				Jac [IRow, ICol] = 1
		
			for IRow in range (NumPackVar + 1, NVPackStir):
				ITL = NumPackGeo
			for l in range(0, NumPackCom):
				ICol = (ITL - 1) * NumPackCom + l
				Jac [IRow, ICol] = 1
		
	for ICol in range(NumPackVar + 1, NVPackStir):
		Jac [IRow, ICol] = 1 
	
	return np.array(Jac)
		
def SciKits():
	from scikits.odes import ode
	
	TimeBegin = ZReadTBEG
	TimeDelta = ZReadTDLT
	TimeFinal = ZReadTEND
	
	tbeg = int(TimeBegin)
	tfin = int(TimeFinal)
	
	NTimeStep = (TimeFinal - TimeBegin) / TimeDelta
	NTS = int(NTimeStep)
	tt = np.linspace(TimeBegin, TimeFinal, NTS)

	solver = ode('cvode', PBRFun)
	TimeUpper = TimeBegin
	
	while TimeUpper <= TimeFinal:
		
		TimeLower = TimeUpper
		TimeUpper = TimeLower + TimeDelta
		TimeMedia = (TimeLower + TimeUpper) * 0.5
		TimeSpan = [TimeLower, TimeMedia, TimeUpper]
		print(TimeSpan)
		PBRNorm()
		GODEVar = PBRVar()
		
		result = solver.solve([TimeLower, TimeMedia, TimeUpper], GODEVar)
		
		print(result)
		PBRSoln()
	
	return()
		
def PBRTime (ProbData):
	# set up the ODE solver for the DAEs
	
	AbsTol = np.zeros((ProbData["NVPackStir"],))
	GODEUpl = np.zeros((ProbData["NVPackStir"],))
	
	NTimeStep = (ProbData["TEND"] - ProbData["TBEG"]) / ProbData["TDLT"]
	NTS = int(NTimeStep)
	TimeUpper = ProbData["TBEG"]

	# print(TimeSpans)
	PBRNorm(ProbData)
	PBRVar(ProbData)
	ODEInit = ProbData["GODEVar"]
	
	tt = np.linspace(ProbData["TBEG"], ProbData["TEND"], NTS)
	
	dvode = scipy.integrate.ode(PBRFun)
	dvode.set_integrator('vode', method='bdf', rtol = ProbData["RTol"], atol = ProbData["ATol"], order=5, with_jacobian=True, nsteps = 9000000)
	dvode.set_initial_value(ODEInit, 0)
	dvode.set_f_params(ProbData)
	solution = np.hstack((0, ODEInit))
	"""
	for t in tt:
		TimeUpper = t
		if not dvode.successful():
			raise ArithmeticError("DVODE step unsuccessful!")
		dvode.integrate(t)
		solution = np.vstack((solution, np.hstack((dvode.t, dvode.y))))
	
	
	"""
	while dvode.successful() and TimeUpper <= ProbData["TEND"]:
		TimeLower = TimeUpper
		TimeUpper = TimeLower + ProbData["TDLT"]
		TimeMedia = (TimeLower + TimeUpper) * 0.5
		TimeSpan = [TimeLower, TimeMedia, TimeUpper]
		print(TimeSpan)
		
		PBRNorm(ProbData)
		PBRVar(ProbData)
		ODEInit = ProbData["GODEVar"]
		dvode.set_initial_value(ODEInit)
		PBRNorm(ProbData)
		if not dvode.successful():
			raise ArithmeticError("DVODE step unsuccessful!")
		dvode.integrate(TimeUpper)
		#PBRSoln()
	
	#PBRWrite()
	return solution
	
def assimulo(ProbData):
	import assimulo.problem
	import assimulo.solvers
	NTimeStep = (ProbData["TEND"] - ProbData["TBEG"]) / ProbData["TDLT"]
	NTS = int(NTimeStep)
	TimeUpper = ProbData["TBEG"]
	
	def rhs(t, y):
		ydot = PBRFun(t,y,ProbData)
		return ydot
	
	PBRNorm(ProbData)
	PBRVar(ProbData)
	
	# Set up the integrator
	batchProblem = assimulo.problem.Explicit_Problem(rhs, ProbData["GODEVar"],0)
	cvode = assimulo.solvers.CVode(batchProblem)
	#cvode = assimulo.solvers.Radau5ODE(batchProblem)
	cvode.atol = ProbData["ATol"]
	cvode.rtol = ProbData["RTol"]
	cvode.maxsteps = 10000000
	cvode.inith = 1e-9
	cvode.discr = "BDF"
	cvode.iter = "Newton"
	tt = np.linspace(ProbData["TBEG"], ProbData["TEND"], NTS)
	t_max = tt[-1]
	n_pts = len(tt)
	cvode_t, cvode_y = cvode.simulate(t_max, n_pts)
	solution = numpy.hstack((
	numpy.asarray([cvode_t]).transpose(),
	numpy.asarray(cvode_y)))
	
		
	return solution
	
def sundials():
	TimeBegin = ZReadTBEG
	TimeDelta = ZReadTDLT
	TimeFinal = ZReadTEND

	AbsTol = np.zeros((NVPackStir,))
	GODEUpl = np.zeros((NVPackStir,))
	
	AbsTol = ZReadATol
	RelTol = ZReadRTol
	
	NTimeStep = (TimeFinal - TimeBegin) / TimeDelta
	NTS = int(NTimeStep)
	TimeUpper = TimeBegin
	
	solver = CVodeSolver(RHS = PBRFun, ROOT = root, SW = [False], lmm = "bdf", iter = "newton", mxsteps=5000, abstol = AbsTol, reltol = RelTol)
	
	while TimeUpper <= TimeFinal:
		TimeLower = TimeUpper
		TimeUpper = TimeLower + TimeDelta
		TimeMedia = (TimeLower + TimeUpper) * 0.5
		TimeSpan = [TimeLower, TimeMedia, TimeUpper]
		print(TimeSpan)
		
		PBRNorm()
		GODEVar = PBRVar()
		solver.init(TimeLower, GODEVar)
		#iter = solver.iter(TimeLower, TimeUpper)
		while True:
			try:
				t, y = solver.step(TimeUpper)
				#t, y = next(iter)
			except CVodeRootException, info:
				if abs(info.y[0] < 0.01):
					solver.SW[0] = True
				solver.init(info.t, [-0.7*info.y[0], 0])
				PBRSoln()
				
			if t > TimeUpper:
				break
	
	return
	
def DASSL(ProbData):	
	from pydas.dassl import DASSL
	PBRNorm(ProbData)
	PBRVar(ProbData)
	NTimeStep = (ProbData["TEND"] - ProbData["TBEG"]) / ProbData["TDLT"]
	NTS = int(NTimeStep)
	TimeUpper = ProbData["TBEG"]
	
	class Problem(DASSL):
		def residual(self, t, y, dydt):
			res = np.asarray(dydt) - \
				PBRFun(t,y,ProbData)
			return res, 0
	# Set up the integrator
	dassl = Problem()
	dassl.initialize(0, ProbData["GODEVar"],
	PBRFun(0, ProbData["GODEVar"], ProbData),
	atol=ProbData["ATol"], rtol=ProbData["RTol"])
	# Carry out the main integration loop
	solution = np.hstack((0, np.asarray(ProbData["GODEVar"],
	)))
	tt = np.linspace(ProbData["TBEG"], ProbData["TEND"], NTS)
	t_max = tt[-1]
	
	#while dassl.t < t_max:
	# dassl.step(t_max)
	# solution = numpy.vstack((solution, numpy.hstack((dassl.t, dassl.y))))
	for t in tt[1:]:
		print(t)
		dassl.advance(t)
		solution = numpy.vstack((solution, numpy.hstack((dassl.t, dassl.y))))
	
	return solution

def PBRTimeLife (ProbData):
	
	import assimulo.problem
	import assimulo.solvers
	
	def rhs(t, y):
		ydot = PBRFun(t,y,ProbData)
		#print(ydot)
		return ydot
	
	TimeBegin = (ProbData["TBEG"])
	TimeFinal = (ProbData["TEND"])
	TimeDelta = (ProbData["TDLT"])
	
	Jac = jaceqn(ProbData)
	
	NTimeStep = (TimeFinal - TimeBegin) / TimeDelta
	NTS = int(NTimeStep)
		
	tt = np.linspace(ProbData["TBEG"], ProbData["TEND"], NTS)
	
	AbsTol = np.zeros((ProbData["NVPackStir"],))
	GODEUpl = np.zeros((ProbData["NVPackStir"],))
	
	PBRNorm(ProbData)
	PBRVar(ProbData)
	
	PBR_prob = assimulo.problem.Explicit_Problem(rhs,ProbData["GODEVar"],0)
	cvode = assimulo.solvers.CVode(PBR_prob)
	cvode.atol = ProbData["ATol"]
	cvode.rtol = ProbData["RTol"]
	cvode.maxsteps = 10000000
	cvode.inith = 1e-9
	cvode.discr = "BDF"
	cvode.iter = "Newton"
	t_max = TimeFinal
	n_pts = NTS
	cvode_t, cvode_y = cvode.simulate(t_max,0,tt)
	solution = numpy.hstack((numpy.asarray([cvode_t]).transpose(),numpy.asarray(cvode_y)))
	
	return solution
	
def Driver():	

	ProbData = PBRFile()
	PBRInitDr (ProbData)
	PBRRead (ProbData)
	solution = PBRTime (ProbData)
	print(solution)
	#GODEFun = PBRFun()
	#solution = PBRTime(ProbData)
	#SciKits()
	#solution = DASSL(ProbData)
	PBRSoln(ProbData)

	return
	
Driver()
