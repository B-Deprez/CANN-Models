Ç
­ý
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
,
Exp
x"T
y"T"
Ttype:

2
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
?
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
¥
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
¾
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
-
Tanh
x"T
y"T"
Ttype:

2

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718
~
CovEmb/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*"
shared_nameCovEmb/embeddings
w
%CovEmb/embeddings/Read/ReadVariableOpReadVariableOpCovEmb/embeddings*
_output_shapes

:*
dtype0
~
SexEmb/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*"
shared_nameSexEmb/embeddings
w
%SexEmb/embeddings/Read/ReadVariableOpReadVariableOpSexEmb/embeddings*
_output_shapes

:*
dtype0

FuelEmb/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*#
shared_nameFuelEmb/embeddings
y
&FuelEmb/embeddings/Read/ReadVariableOpReadVariableOpFuelEmb/embeddings*
_output_shapes

:*
dtype0

UsageEmb/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*$
shared_nameUsageEmb/embeddings
{
'UsageEmb/embeddings/Read/ReadVariableOpReadVariableOpUsageEmb/embeddings*
_output_shapes

:*
dtype0

FleetEmb/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*$
shared_nameFleetEmb/embeddings
{
'FleetEmb/embeddings/Read/ReadVariableOpReadVariableOpFleetEmb/embeddings*
_output_shapes

:*
dtype0
|
PcEmb/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:P*!
shared_namePcEmb/embeddings
u
$PcEmb/embeddings/Read/ReadVariableOpReadVariableOpPcEmb/embeddings*
_output_shapes

:P*
dtype0

batch_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_4/gamma

/batch_normalization_4/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_4/gamma*
_output_shapes
:*
dtype0

batch_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_4/beta

.batch_normalization_4/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_4/beta*
_output_shapes
:*
dtype0

!batch_normalization_4/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_4/moving_mean

5batch_normalization_4/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_4/moving_mean*
_output_shapes
:*
dtype0
¢
%batch_normalization_4/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_4/moving_variance

9batch_normalization_4/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_4/moving_variance*
_output_shapes
:*
dtype0
x
hidden1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namehidden1/kernel
q
"hidden1/kernel/Read/ReadVariableOpReadVariableOphidden1/kernel*
_output_shapes

:*
dtype0
p
hidden1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namehidden1/bias
i
 hidden1/bias/Read/ReadVariableOpReadVariableOphidden1/bias*
_output_shapes
:*
dtype0
x
hidden2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*
shared_namehidden2/kernel
q
"hidden2/kernel/Read/ReadVariableOpReadVariableOphidden2/kernel*
_output_shapes

:
*
dtype0
p
hidden2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namehidden2/bias
i
 hidden2/bias/Read/ReadVariableOpReadVariableOphidden2/bias*
_output_shapes
:
*
dtype0
x
hidden3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*
shared_namehidden3/kernel
q
"hidden3/kernel/Read/ReadVariableOpReadVariableOphidden3/kernel*
_output_shapes

:
*
dtype0
p
hidden3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namehidden3/bias
i
 hidden3/bias/Read/ReadVariableOpReadVariableOphidden3/bias*
_output_shapes
:*
dtype0
x
Network/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_nameNetwork/kernel
q
"Network/kernel/Read/ReadVariableOpReadVariableOpNetwork/kernel*
_output_shapes

:*
dtype0
p
Network/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameNetwork/bias
i
 Network/bias/Read/ReadVariableOpReadVariableOpNetwork/bias*
_output_shapes
:*
dtype0
z
Response/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_nameResponse/kernel
s
#Response/kernel/Read/ReadVariableOpReadVariableOpResponse/kernel*
_output_shapes

:*
dtype0
r
Response/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameResponse/bias
k
!Response/bias/Read/ReadVariableOpReadVariableOpResponse/bias*
_output_shapes
:*
dtype0
h

Nadam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
Nadam/iter
a
Nadam/iter/Read/ReadVariableOpReadVariableOp
Nadam/iter*
_output_shapes
: *
dtype0	
l
Nadam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameNadam/beta_1
e
 Nadam/beta_1/Read/ReadVariableOpReadVariableOpNadam/beta_1*
_output_shapes
: *
dtype0
l
Nadam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameNadam/beta_2
e
 Nadam/beta_2/Read/ReadVariableOpReadVariableOpNadam/beta_2*
_output_shapes
: *
dtype0
j
Nadam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameNadam/decay
c
Nadam/decay/Read/ReadVariableOpReadVariableOpNadam/decay*
_output_shapes
: *
dtype0
z
Nadam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameNadam/learning_rate
s
'Nadam/learning_rate/Read/ReadVariableOpReadVariableOpNadam/learning_rate*
_output_shapes
: *
dtype0
|
Nadam/momentum_cacheVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameNadam/momentum_cache
u
(Nadam/momentum_cache/Read/ReadVariableOpReadVariableOpNadam/momentum_cache*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0

Nadam/CovEmb/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:**
shared_nameNadam/CovEmb/embeddings/m

-Nadam/CovEmb/embeddings/m/Read/ReadVariableOpReadVariableOpNadam/CovEmb/embeddings/m*
_output_shapes

:*
dtype0

Nadam/SexEmb/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:**
shared_nameNadam/SexEmb/embeddings/m

-Nadam/SexEmb/embeddings/m/Read/ReadVariableOpReadVariableOpNadam/SexEmb/embeddings/m*
_output_shapes

:*
dtype0

Nadam/FuelEmb/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*+
shared_nameNadam/FuelEmb/embeddings/m

.Nadam/FuelEmb/embeddings/m/Read/ReadVariableOpReadVariableOpNadam/FuelEmb/embeddings/m*
_output_shapes

:*
dtype0

Nadam/UsageEmb/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*,
shared_nameNadam/UsageEmb/embeddings/m

/Nadam/UsageEmb/embeddings/m/Read/ReadVariableOpReadVariableOpNadam/UsageEmb/embeddings/m*
_output_shapes

:*
dtype0

Nadam/FleetEmb/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*,
shared_nameNadam/FleetEmb/embeddings/m

/Nadam/FleetEmb/embeddings/m/Read/ReadVariableOpReadVariableOpNadam/FleetEmb/embeddings/m*
_output_shapes

:*
dtype0

Nadam/PcEmb/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:P*)
shared_nameNadam/PcEmb/embeddings/m

,Nadam/PcEmb/embeddings/m/Read/ReadVariableOpReadVariableOpNadam/PcEmb/embeddings/m*
_output_shapes

:P*
dtype0

#Nadam/batch_normalization_4/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Nadam/batch_normalization_4/gamma/m

7Nadam/batch_normalization_4/gamma/m/Read/ReadVariableOpReadVariableOp#Nadam/batch_normalization_4/gamma/m*
_output_shapes
:*
dtype0

"Nadam/batch_normalization_4/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Nadam/batch_normalization_4/beta/m

6Nadam/batch_normalization_4/beta/m/Read/ReadVariableOpReadVariableOp"Nadam/batch_normalization_4/beta/m*
_output_shapes
:*
dtype0

Nadam/hidden1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameNadam/hidden1/kernel/m

*Nadam/hidden1/kernel/m/Read/ReadVariableOpReadVariableOpNadam/hidden1/kernel/m*
_output_shapes

:*
dtype0

Nadam/hidden1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameNadam/hidden1/bias/m
y
(Nadam/hidden1/bias/m/Read/ReadVariableOpReadVariableOpNadam/hidden1/bias/m*
_output_shapes
:*
dtype0

Nadam/hidden2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*'
shared_nameNadam/hidden2/kernel/m

*Nadam/hidden2/kernel/m/Read/ReadVariableOpReadVariableOpNadam/hidden2/kernel/m*
_output_shapes

:
*
dtype0

Nadam/hidden2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*%
shared_nameNadam/hidden2/bias/m
y
(Nadam/hidden2/bias/m/Read/ReadVariableOpReadVariableOpNadam/hidden2/bias/m*
_output_shapes
:
*
dtype0

Nadam/hidden3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*'
shared_nameNadam/hidden3/kernel/m

*Nadam/hidden3/kernel/m/Read/ReadVariableOpReadVariableOpNadam/hidden3/kernel/m*
_output_shapes

:
*
dtype0

Nadam/hidden3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameNadam/hidden3/bias/m
y
(Nadam/hidden3/bias/m/Read/ReadVariableOpReadVariableOpNadam/hidden3/bias/m*
_output_shapes
:*
dtype0

Nadam/Network/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameNadam/Network/kernel/m

*Nadam/Network/kernel/m/Read/ReadVariableOpReadVariableOpNadam/Network/kernel/m*
_output_shapes

:*
dtype0

Nadam/Network/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameNadam/Network/bias/m
y
(Nadam/Network/bias/m/Read/ReadVariableOpReadVariableOpNadam/Network/bias/m*
_output_shapes
:*
dtype0

Nadam/CovEmb/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:**
shared_nameNadam/CovEmb/embeddings/v

-Nadam/CovEmb/embeddings/v/Read/ReadVariableOpReadVariableOpNadam/CovEmb/embeddings/v*
_output_shapes

:*
dtype0

Nadam/SexEmb/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:**
shared_nameNadam/SexEmb/embeddings/v

-Nadam/SexEmb/embeddings/v/Read/ReadVariableOpReadVariableOpNadam/SexEmb/embeddings/v*
_output_shapes

:*
dtype0

Nadam/FuelEmb/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*+
shared_nameNadam/FuelEmb/embeddings/v

.Nadam/FuelEmb/embeddings/v/Read/ReadVariableOpReadVariableOpNadam/FuelEmb/embeddings/v*
_output_shapes

:*
dtype0

Nadam/UsageEmb/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*,
shared_nameNadam/UsageEmb/embeddings/v

/Nadam/UsageEmb/embeddings/v/Read/ReadVariableOpReadVariableOpNadam/UsageEmb/embeddings/v*
_output_shapes

:*
dtype0

Nadam/FleetEmb/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*,
shared_nameNadam/FleetEmb/embeddings/v

/Nadam/FleetEmb/embeddings/v/Read/ReadVariableOpReadVariableOpNadam/FleetEmb/embeddings/v*
_output_shapes

:*
dtype0

Nadam/PcEmb/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:P*)
shared_nameNadam/PcEmb/embeddings/v

,Nadam/PcEmb/embeddings/v/Read/ReadVariableOpReadVariableOpNadam/PcEmb/embeddings/v*
_output_shapes

:P*
dtype0

#Nadam/batch_normalization_4/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Nadam/batch_normalization_4/gamma/v

7Nadam/batch_normalization_4/gamma/v/Read/ReadVariableOpReadVariableOp#Nadam/batch_normalization_4/gamma/v*
_output_shapes
:*
dtype0

"Nadam/batch_normalization_4/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Nadam/batch_normalization_4/beta/v

6Nadam/batch_normalization_4/beta/v/Read/ReadVariableOpReadVariableOp"Nadam/batch_normalization_4/beta/v*
_output_shapes
:*
dtype0

Nadam/hidden1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameNadam/hidden1/kernel/v

*Nadam/hidden1/kernel/v/Read/ReadVariableOpReadVariableOpNadam/hidden1/kernel/v*
_output_shapes

:*
dtype0

Nadam/hidden1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameNadam/hidden1/bias/v
y
(Nadam/hidden1/bias/v/Read/ReadVariableOpReadVariableOpNadam/hidden1/bias/v*
_output_shapes
:*
dtype0

Nadam/hidden2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*'
shared_nameNadam/hidden2/kernel/v

*Nadam/hidden2/kernel/v/Read/ReadVariableOpReadVariableOpNadam/hidden2/kernel/v*
_output_shapes

:
*
dtype0

Nadam/hidden2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*%
shared_nameNadam/hidden2/bias/v
y
(Nadam/hidden2/bias/v/Read/ReadVariableOpReadVariableOpNadam/hidden2/bias/v*
_output_shapes
:
*
dtype0

Nadam/hidden3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*'
shared_nameNadam/hidden3/kernel/v

*Nadam/hidden3/kernel/v/Read/ReadVariableOpReadVariableOpNadam/hidden3/kernel/v*
_output_shapes

:
*
dtype0

Nadam/hidden3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameNadam/hidden3/bias/v
y
(Nadam/hidden3/bias/v/Read/ReadVariableOpReadVariableOpNadam/hidden3/bias/v*
_output_shapes
:*
dtype0

Nadam/Network/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameNadam/Network/kernel/v

*Nadam/Network/kernel/v/Read/ReadVariableOpReadVariableOpNadam/Network/kernel/v*
_output_shapes

:*
dtype0

Nadam/Network/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameNadam/Network/bias/v
y
(Nadam/Network/bias/v/Read/ReadVariableOpReadVariableOpNadam/Network/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
Óv
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*v
valuevBv Búu
©
layer-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer_with_weights-0
layer-6
layer_with_weights-1
layer-7
	layer_with_weights-2
	layer-8

layer_with_weights-3

layer-9
layer_with_weights-4
layer-10
layer_with_weights-5
layer-11
layer-12
layer-13
layer-14
layer-15
layer-16
layer-17
layer-18
layer-19
layer_with_weights-6
layer-20
layer_with_weights-7
layer-21
layer_with_weights-8
layer-22
layer_with_weights-9
layer-23
layer_with_weights-10
layer-24
layer-25
layer-26
layer_with_weights-11
layer-27
	optimizer
	variables
trainable_variables
 regularization_losses
!	keras_api
"
signatures
 
 
 
 
 
 
b
#
embeddings
$trainable_variables
%	variables
&regularization_losses
'	keras_api
b
(
embeddings
)trainable_variables
*	variables
+regularization_losses
,	keras_api
b
-
embeddings
.trainable_variables
/	variables
0regularization_losses
1	keras_api
b
2
embeddings
3trainable_variables
4	variables
5regularization_losses
6	keras_api
b
7
embeddings
8trainable_variables
9	variables
:regularization_losses
;	keras_api
b
<
embeddings
=trainable_variables
>	variables
?regularization_losses
@	keras_api
 
R
Atrainable_variables
B	variables
Cregularization_losses
D	keras_api
R
Etrainable_variables
F	variables
Gregularization_losses
H	keras_api
R
Itrainable_variables
J	variables
Kregularization_losses
L	keras_api
R
Mtrainable_variables
N	variables
Oregularization_losses
P	keras_api
R
Qtrainable_variables
R	variables
Sregularization_losses
T	keras_api
R
Utrainable_variables
V	variables
Wregularization_losses
X	keras_api
R
Ytrainable_variables
Z	variables
[regularization_losses
\	keras_api

]axis
	^gamma
_beta
`moving_mean
amoving_variance
btrainable_variables
c	variables
dregularization_losses
e	keras_api
h

fkernel
gbias
htrainable_variables
i	variables
jregularization_losses
k	keras_api
h

lkernel
mbias
ntrainable_variables
o	variables
pregularization_losses
q	keras_api
h

rkernel
sbias
ttrainable_variables
u	variables
vregularization_losses
w	keras_api
h

xkernel
ybias
ztrainable_variables
{	variables
|regularization_losses
}	keras_api
 
T
~trainable_variables
	variables
regularization_losses
	keras_api
n
kernel
	bias
trainable_variables
	variables
regularization_losses
	keras_api

	iter
beta_1
beta_2

decay
learning_rate
momentum_cache#mü(mý-mþ2mÿ7m<m^m_mfmgmlmmmrmsmxmym#v(v-v2v7v<v^v_vfvgvlvmvrvsvxvyv

#0
(1
-2
23
74
<5
^6
_7
`8
a9
f10
g11
l12
m13
r14
s15
x16
y17
18
19
v
#0
(1
-2
23
74
<5
^6
_7
f8
g9
l10
m11
r12
s13
x14
y15
 
²
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
 regularization_losses
metrics
non_trainable_variables
layers
 
a_
VARIABLE_VALUECovEmb/embeddings:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUE

#0

#0
 
²
 layer_regularization_losses
layer_metrics
$trainable_variables
%	variables
&regularization_losses
metrics
non_trainable_variables
layers
a_
VARIABLE_VALUESexEmb/embeddings:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUE

(0

(0
 
²
 layer_regularization_losses
layer_metrics
)trainable_variables
*	variables
+regularization_losses
metrics
non_trainable_variables
layers
b`
VARIABLE_VALUEFuelEmb/embeddings:layer_with_weights-2/embeddings/.ATTRIBUTES/VARIABLE_VALUE

-0

-0
 
²
 layer_regularization_losses
layer_metrics
.trainable_variables
/	variables
0regularization_losses
metrics
 non_trainable_variables
¡layers
ca
VARIABLE_VALUEUsageEmb/embeddings:layer_with_weights-3/embeddings/.ATTRIBUTES/VARIABLE_VALUE

20

20
 
²
 ¢layer_regularization_losses
£layer_metrics
3trainable_variables
4	variables
5regularization_losses
¤metrics
¥non_trainable_variables
¦layers
ca
VARIABLE_VALUEFleetEmb/embeddings:layer_with_weights-4/embeddings/.ATTRIBUTES/VARIABLE_VALUE

70

70
 
²
 §layer_regularization_losses
¨layer_metrics
8trainable_variables
9	variables
:regularization_losses
©metrics
ªnon_trainable_variables
«layers
`^
VARIABLE_VALUEPcEmb/embeddings:layer_with_weights-5/embeddings/.ATTRIBUTES/VARIABLE_VALUE

<0

<0
 
²
 ¬layer_regularization_losses
­layer_metrics
=trainable_variables
>	variables
?regularization_losses
®metrics
¯non_trainable_variables
°layers
 
 
 
²
 ±layer_regularization_losses
²layer_metrics
Atrainable_variables
B	variables
Cregularization_losses
³metrics
´non_trainable_variables
µlayers
 
 
 
²
 ¶layer_regularization_losses
·layer_metrics
Etrainable_variables
F	variables
Gregularization_losses
¸metrics
¹non_trainable_variables
ºlayers
 
 
 
²
 »layer_regularization_losses
¼layer_metrics
Itrainable_variables
J	variables
Kregularization_losses
½metrics
¾non_trainable_variables
¿layers
 
 
 
²
 Àlayer_regularization_losses
Álayer_metrics
Mtrainable_variables
N	variables
Oregularization_losses
Âmetrics
Ãnon_trainable_variables
Älayers
 
 
 
²
 Ålayer_regularization_losses
Ælayer_metrics
Qtrainable_variables
R	variables
Sregularization_losses
Çmetrics
Ènon_trainable_variables
Élayers
 
 
 
²
 Êlayer_regularization_losses
Ëlayer_metrics
Utrainable_variables
V	variables
Wregularization_losses
Ìmetrics
Ínon_trainable_variables
Îlayers
 
 
 
²
 Ïlayer_regularization_losses
Ðlayer_metrics
Ytrainable_variables
Z	variables
[regularization_losses
Ñmetrics
Ònon_trainable_variables
Ólayers
 
fd
VARIABLE_VALUEbatch_normalization_4/gamma5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_4/beta4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_4/moving_mean;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_4/moving_variance?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

^0
_1

^0
_1
`2
a3
 
²
 Ôlayer_regularization_losses
Õlayer_metrics
btrainable_variables
c	variables
dregularization_losses
Ömetrics
×non_trainable_variables
Ølayers
ZX
VARIABLE_VALUEhidden1/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEhidden1/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

f0
g1

f0
g1
 
²
 Ùlayer_regularization_losses
Úlayer_metrics
htrainable_variables
i	variables
jregularization_losses
Ûmetrics
Ünon_trainable_variables
Ýlayers
ZX
VARIABLE_VALUEhidden2/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEhidden2/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

l0
m1

l0
m1
 
²
 Þlayer_regularization_losses
ßlayer_metrics
ntrainable_variables
o	variables
pregularization_losses
àmetrics
ánon_trainable_variables
âlayers
ZX
VARIABLE_VALUEhidden3/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEhidden3/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE

r0
s1

r0
s1
 
²
 ãlayer_regularization_losses
älayer_metrics
ttrainable_variables
u	variables
vregularization_losses
åmetrics
ænon_trainable_variables
çlayers
[Y
VARIABLE_VALUENetwork/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUENetwork/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE

x0
y1

x0
y1
 
²
 èlayer_regularization_losses
élayer_metrics
ztrainable_variables
{	variables
|regularization_losses
êmetrics
ënon_trainable_variables
ìlayers
 
 
 
³
 ílayer_regularization_losses
îlayer_metrics
~trainable_variables
	variables
regularization_losses
ïmetrics
ðnon_trainable_variables
ñlayers
\Z
VARIABLE_VALUEResponse/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEResponse/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1
 
µ
 òlayer_regularization_losses
ólayer_metrics
trainable_variables
	variables
regularization_losses
ômetrics
õnon_trainable_variables
ölayers
IG
VARIABLE_VALUE
Nadam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUENadam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUENadam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUENadam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUENadam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUENadam/momentum_cache3optimizer/momentum_cache/.ATTRIBUTES/VARIABLE_VALUE
 
 

÷0

`0
a1
2
3
Ö
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

`0
a1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

0
1
 
8

øtotal

ùcount
ú	variables
û	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

ø0
ù1

ú	variables

VARIABLE_VALUENadam/CovEmb/embeddings/mVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUENadam/SexEmb/embeddings/mVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUENadam/FuelEmb/embeddings/mVlayer_with_weights-2/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUENadam/UsageEmb/embeddings/mVlayer_with_weights-3/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUENadam/FleetEmb/embeddings/mVlayer_with_weights-4/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUENadam/PcEmb/embeddings/mVlayer_with_weights-5/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Nadam/batch_normalization_4/gamma/mQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Nadam/batch_normalization_4/beta/mPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUENadam/hidden1/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUENadam/hidden1/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUENadam/hidden2/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUENadam/hidden2/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUENadam/hidden3/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUENadam/hidden3/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUENadam/Network/kernel/mSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUENadam/Network/bias/mQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUENadam/CovEmb/embeddings/vVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUENadam/SexEmb/embeddings/vVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUENadam/FuelEmb/embeddings/vVlayer_with_weights-2/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUENadam/UsageEmb/embeddings/vVlayer_with_weights-3/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUENadam/FleetEmb/embeddings/vVlayer_with_weights-4/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUENadam/PcEmb/embeddings/vVlayer_with_weights-5/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Nadam/batch_normalization_4/gamma/vQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Nadam/batch_normalization_4/beta/vPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUENadam/hidden1/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUENadam/hidden1/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUENadam/hidden2/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUENadam/hidden2/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUENadam/hidden3/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUENadam/hidden3/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUENadam/Network/kernel/vSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUENadam/Network/bias/vQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{
serving_default_CoveragePlaceholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
y
serving_default_DesignPlaceholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
x
serving_default_FleetPlaceholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
w
serving_default_FuelPlaceholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
y
serving_default_LogGAMPlaceholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
}
serving_default_PostalCodePlaceholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
v
serving_default_SexPlaceholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
x
serving_default_UsagePlaceholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ

StatefulPartitionedCallStatefulPartitionedCallserving_default_Coverageserving_default_Designserving_default_Fleetserving_default_Fuelserving_default_LogGAMserving_default_PostalCodeserving_default_Sexserving_default_UsagePcEmb/embeddingsFleetEmb/embeddingsUsageEmb/embeddingsFuelEmb/embeddingsSexEmb/embeddingsCovEmb/embeddings%batch_normalization_4/moving_variancebatch_normalization_4/gamma!batch_normalization_4/moving_meanbatch_normalization_4/betahidden1/kernelhidden1/biashidden2/kernelhidden2/biashidden3/kernelhidden3/biasNetwork/kernelNetwork/biasResponse/kernelResponse/bias*'
Tin 
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference_signature_wrapper_18241
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ó
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename%CovEmb/embeddings/Read/ReadVariableOp%SexEmb/embeddings/Read/ReadVariableOp&FuelEmb/embeddings/Read/ReadVariableOp'UsageEmb/embeddings/Read/ReadVariableOp'FleetEmb/embeddings/Read/ReadVariableOp$PcEmb/embeddings/Read/ReadVariableOp/batch_normalization_4/gamma/Read/ReadVariableOp.batch_normalization_4/beta/Read/ReadVariableOp5batch_normalization_4/moving_mean/Read/ReadVariableOp9batch_normalization_4/moving_variance/Read/ReadVariableOp"hidden1/kernel/Read/ReadVariableOp hidden1/bias/Read/ReadVariableOp"hidden2/kernel/Read/ReadVariableOp hidden2/bias/Read/ReadVariableOp"hidden3/kernel/Read/ReadVariableOp hidden3/bias/Read/ReadVariableOp"Network/kernel/Read/ReadVariableOp Network/bias/Read/ReadVariableOp#Response/kernel/Read/ReadVariableOp!Response/bias/Read/ReadVariableOpNadam/iter/Read/ReadVariableOp Nadam/beta_1/Read/ReadVariableOp Nadam/beta_2/Read/ReadVariableOpNadam/decay/Read/ReadVariableOp'Nadam/learning_rate/Read/ReadVariableOp(Nadam/momentum_cache/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp-Nadam/CovEmb/embeddings/m/Read/ReadVariableOp-Nadam/SexEmb/embeddings/m/Read/ReadVariableOp.Nadam/FuelEmb/embeddings/m/Read/ReadVariableOp/Nadam/UsageEmb/embeddings/m/Read/ReadVariableOp/Nadam/FleetEmb/embeddings/m/Read/ReadVariableOp,Nadam/PcEmb/embeddings/m/Read/ReadVariableOp7Nadam/batch_normalization_4/gamma/m/Read/ReadVariableOp6Nadam/batch_normalization_4/beta/m/Read/ReadVariableOp*Nadam/hidden1/kernel/m/Read/ReadVariableOp(Nadam/hidden1/bias/m/Read/ReadVariableOp*Nadam/hidden2/kernel/m/Read/ReadVariableOp(Nadam/hidden2/bias/m/Read/ReadVariableOp*Nadam/hidden3/kernel/m/Read/ReadVariableOp(Nadam/hidden3/bias/m/Read/ReadVariableOp*Nadam/Network/kernel/m/Read/ReadVariableOp(Nadam/Network/bias/m/Read/ReadVariableOp-Nadam/CovEmb/embeddings/v/Read/ReadVariableOp-Nadam/SexEmb/embeddings/v/Read/ReadVariableOp.Nadam/FuelEmb/embeddings/v/Read/ReadVariableOp/Nadam/UsageEmb/embeddings/v/Read/ReadVariableOp/Nadam/FleetEmb/embeddings/v/Read/ReadVariableOp,Nadam/PcEmb/embeddings/v/Read/ReadVariableOp7Nadam/batch_normalization_4/gamma/v/Read/ReadVariableOp6Nadam/batch_normalization_4/beta/v/Read/ReadVariableOp*Nadam/hidden1/kernel/v/Read/ReadVariableOp(Nadam/hidden1/bias/v/Read/ReadVariableOp*Nadam/hidden2/kernel/v/Read/ReadVariableOp(Nadam/hidden2/bias/v/Read/ReadVariableOp*Nadam/hidden3/kernel/v/Read/ReadVariableOp(Nadam/hidden3/bias/v/Read/ReadVariableOp*Nadam/Network/kernel/v/Read/ReadVariableOp(Nadam/Network/bias/v/Read/ReadVariableOpConst*I
TinB
@2>	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *'
f"R 
__inference__traced_save_19157

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameCovEmb/embeddingsSexEmb/embeddingsFuelEmb/embeddingsUsageEmb/embeddingsFleetEmb/embeddingsPcEmb/embeddingsbatch_normalization_4/gammabatch_normalization_4/beta!batch_normalization_4/moving_mean%batch_normalization_4/moving_variancehidden1/kernelhidden1/biashidden2/kernelhidden2/biashidden3/kernelhidden3/biasNetwork/kernelNetwork/biasResponse/kernelResponse/bias
Nadam/iterNadam/beta_1Nadam/beta_2Nadam/decayNadam/learning_rateNadam/momentum_cachetotalcountNadam/CovEmb/embeddings/mNadam/SexEmb/embeddings/mNadam/FuelEmb/embeddings/mNadam/UsageEmb/embeddings/mNadam/FleetEmb/embeddings/mNadam/PcEmb/embeddings/m#Nadam/batch_normalization_4/gamma/m"Nadam/batch_normalization_4/beta/mNadam/hidden1/kernel/mNadam/hidden1/bias/mNadam/hidden2/kernel/mNadam/hidden2/bias/mNadam/hidden3/kernel/mNadam/hidden3/bias/mNadam/Network/kernel/mNadam/Network/bias/mNadam/CovEmb/embeddings/vNadam/SexEmb/embeddings/vNadam/FuelEmb/embeddings/vNadam/UsageEmb/embeddings/vNadam/FleetEmb/embeddings/vNadam/PcEmb/embeddings/v#Nadam/batch_normalization_4/gamma/v"Nadam/batch_normalization_4/beta/vNadam/hidden1/kernel/vNadam/hidden1/bias/vNadam/hidden2/kernel/vNadam/hidden2/bias/vNadam/hidden3/kernel/vNadam/hidden3/bias/vNadam/Network/kernel/vNadam/Network/bias/v*H
TinA
?2=*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__traced_restore_19347î
°
ù%
!__inference__traced_restore_19347
file_prefix4
"assignvariableop_covemb_embeddings:6
$assignvariableop_1_sexemb_embeddings:7
%assignvariableop_2_fuelemb_embeddings:8
&assignvariableop_3_usageemb_embeddings:8
&assignvariableop_4_fleetemb_embeddings:5
#assignvariableop_5_pcemb_embeddings:P<
.assignvariableop_6_batch_normalization_4_gamma:;
-assignvariableop_7_batch_normalization_4_beta:B
4assignvariableop_8_batch_normalization_4_moving_mean:F
8assignvariableop_9_batch_normalization_4_moving_variance:4
"assignvariableop_10_hidden1_kernel:.
 assignvariableop_11_hidden1_bias:4
"assignvariableop_12_hidden2_kernel:
.
 assignvariableop_13_hidden2_bias:
4
"assignvariableop_14_hidden3_kernel:
.
 assignvariableop_15_hidden3_bias:4
"assignvariableop_16_network_kernel:.
 assignvariableop_17_network_bias:5
#assignvariableop_18_response_kernel:/
!assignvariableop_19_response_bias:(
assignvariableop_20_nadam_iter:	 *
 assignvariableop_21_nadam_beta_1: *
 assignvariableop_22_nadam_beta_2: )
assignvariableop_23_nadam_decay: 1
'assignvariableop_24_nadam_learning_rate: 2
(assignvariableop_25_nadam_momentum_cache: #
assignvariableop_26_total: #
assignvariableop_27_count: ?
-assignvariableop_28_nadam_covemb_embeddings_m:?
-assignvariableop_29_nadam_sexemb_embeddings_m:@
.assignvariableop_30_nadam_fuelemb_embeddings_m:A
/assignvariableop_31_nadam_usageemb_embeddings_m:A
/assignvariableop_32_nadam_fleetemb_embeddings_m:>
,assignvariableop_33_nadam_pcemb_embeddings_m:PE
7assignvariableop_34_nadam_batch_normalization_4_gamma_m:D
6assignvariableop_35_nadam_batch_normalization_4_beta_m:<
*assignvariableop_36_nadam_hidden1_kernel_m:6
(assignvariableop_37_nadam_hidden1_bias_m:<
*assignvariableop_38_nadam_hidden2_kernel_m:
6
(assignvariableop_39_nadam_hidden2_bias_m:
<
*assignvariableop_40_nadam_hidden3_kernel_m:
6
(assignvariableop_41_nadam_hidden3_bias_m:<
*assignvariableop_42_nadam_network_kernel_m:6
(assignvariableop_43_nadam_network_bias_m:?
-assignvariableop_44_nadam_covemb_embeddings_v:?
-assignvariableop_45_nadam_sexemb_embeddings_v:@
.assignvariableop_46_nadam_fuelemb_embeddings_v:A
/assignvariableop_47_nadam_usageemb_embeddings_v:A
/assignvariableop_48_nadam_fleetemb_embeddings_v:>
,assignvariableop_49_nadam_pcemb_embeddings_v:PE
7assignvariableop_50_nadam_batch_normalization_4_gamma_v:D
6assignvariableop_51_nadam_batch_normalization_4_beta_v:<
*assignvariableop_52_nadam_hidden1_kernel_v:6
(assignvariableop_53_nadam_hidden1_bias_v:<
*assignvariableop_54_nadam_hidden2_kernel_v:
6
(assignvariableop_55_nadam_hidden2_bias_v:
<
*assignvariableop_56_nadam_hidden3_kernel_v:
6
(assignvariableop_57_nadam_hidden3_bias_v:<
*assignvariableop_58_nadam_network_kernel_v:6
(assignvariableop_59_nadam_network_bias_v:
identity_61¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_48¢AssignVariableOp_49¢AssignVariableOp_5¢AssignVariableOp_50¢AssignVariableOp_51¢AssignVariableOp_52¢AssignVariableOp_53¢AssignVariableOp_54¢AssignVariableOp_55¢AssignVariableOp_56¢AssignVariableOp_57¢AssignVariableOp_58¢AssignVariableOp_59¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9È"
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:=*
dtype0*Ô!
valueÊ!BÇ!=B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-2/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-3/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-4/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-5/embeddings/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/momentum_cache/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-2/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-3/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-4/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-5/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-2/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-3/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-4/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-5/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:=*
dtype0*
valueB=B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesß
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapes÷
ô:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*K
dtypesA
?2=	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity¡
AssignVariableOpAssignVariableOp"assignvariableop_covemb_embeddingsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1©
AssignVariableOp_1AssignVariableOp$assignvariableop_1_sexemb_embeddingsIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2ª
AssignVariableOp_2AssignVariableOp%assignvariableop_2_fuelemb_embeddingsIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3«
AssignVariableOp_3AssignVariableOp&assignvariableop_3_usageemb_embeddingsIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4«
AssignVariableOp_4AssignVariableOp&assignvariableop_4_fleetemb_embeddingsIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5¨
AssignVariableOp_5AssignVariableOp#assignvariableop_5_pcemb_embeddingsIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6³
AssignVariableOp_6AssignVariableOp.assignvariableop_6_batch_normalization_4_gammaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7²
AssignVariableOp_7AssignVariableOp-assignvariableop_7_batch_normalization_4_betaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8¹
AssignVariableOp_8AssignVariableOp4assignvariableop_8_batch_normalization_4_moving_meanIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9½
AssignVariableOp_9AssignVariableOp8assignvariableop_9_batch_normalization_4_moving_varianceIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10ª
AssignVariableOp_10AssignVariableOp"assignvariableop_10_hidden1_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11¨
AssignVariableOp_11AssignVariableOp assignvariableop_11_hidden1_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12ª
AssignVariableOp_12AssignVariableOp"assignvariableop_12_hidden2_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13¨
AssignVariableOp_13AssignVariableOp assignvariableop_13_hidden2_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14ª
AssignVariableOp_14AssignVariableOp"assignvariableop_14_hidden3_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15¨
AssignVariableOp_15AssignVariableOp assignvariableop_15_hidden3_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16ª
AssignVariableOp_16AssignVariableOp"assignvariableop_16_network_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17¨
AssignVariableOp_17AssignVariableOp assignvariableop_17_network_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18«
AssignVariableOp_18AssignVariableOp#assignvariableop_18_response_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19©
AssignVariableOp_19AssignVariableOp!assignvariableop_19_response_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_20¦
AssignVariableOp_20AssignVariableOpassignvariableop_20_nadam_iterIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21¨
AssignVariableOp_21AssignVariableOp assignvariableop_21_nadam_beta_1Identity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22¨
AssignVariableOp_22AssignVariableOp assignvariableop_22_nadam_beta_2Identity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23§
AssignVariableOp_23AssignVariableOpassignvariableop_23_nadam_decayIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24¯
AssignVariableOp_24AssignVariableOp'assignvariableop_24_nadam_learning_rateIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25°
AssignVariableOp_25AssignVariableOp(assignvariableop_25_nadam_momentum_cacheIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26¡
AssignVariableOp_26AssignVariableOpassignvariableop_26_totalIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27¡
AssignVariableOp_27AssignVariableOpassignvariableop_27_countIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28µ
AssignVariableOp_28AssignVariableOp-assignvariableop_28_nadam_covemb_embeddings_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29µ
AssignVariableOp_29AssignVariableOp-assignvariableop_29_nadam_sexemb_embeddings_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30¶
AssignVariableOp_30AssignVariableOp.assignvariableop_30_nadam_fuelemb_embeddings_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31·
AssignVariableOp_31AssignVariableOp/assignvariableop_31_nadam_usageemb_embeddings_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32·
AssignVariableOp_32AssignVariableOp/assignvariableop_32_nadam_fleetemb_embeddings_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33´
AssignVariableOp_33AssignVariableOp,assignvariableop_33_nadam_pcemb_embeddings_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34¿
AssignVariableOp_34AssignVariableOp7assignvariableop_34_nadam_batch_normalization_4_gamma_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35¾
AssignVariableOp_35AssignVariableOp6assignvariableop_35_nadam_batch_normalization_4_beta_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36²
AssignVariableOp_36AssignVariableOp*assignvariableop_36_nadam_hidden1_kernel_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37°
AssignVariableOp_37AssignVariableOp(assignvariableop_37_nadam_hidden1_bias_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38²
AssignVariableOp_38AssignVariableOp*assignvariableop_38_nadam_hidden2_kernel_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39°
AssignVariableOp_39AssignVariableOp(assignvariableop_39_nadam_hidden2_bias_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40²
AssignVariableOp_40AssignVariableOp*assignvariableop_40_nadam_hidden3_kernel_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41°
AssignVariableOp_41AssignVariableOp(assignvariableop_41_nadam_hidden3_bias_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42²
AssignVariableOp_42AssignVariableOp*assignvariableop_42_nadam_network_kernel_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43°
AssignVariableOp_43AssignVariableOp(assignvariableop_43_nadam_network_bias_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44µ
AssignVariableOp_44AssignVariableOp-assignvariableop_44_nadam_covemb_embeddings_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45µ
AssignVariableOp_45AssignVariableOp-assignvariableop_45_nadam_sexemb_embeddings_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46¶
AssignVariableOp_46AssignVariableOp.assignvariableop_46_nadam_fuelemb_embeddings_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47·
AssignVariableOp_47AssignVariableOp/assignvariableop_47_nadam_usageemb_embeddings_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48·
AssignVariableOp_48AssignVariableOp/assignvariableop_48_nadam_fleetemb_embeddings_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49´
AssignVariableOp_49AssignVariableOp,assignvariableop_49_nadam_pcemb_embeddings_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50¿
AssignVariableOp_50AssignVariableOp7assignvariableop_50_nadam_batch_normalization_4_gamma_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51¾
AssignVariableOp_51AssignVariableOp6assignvariableop_51_nadam_batch_normalization_4_beta_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52²
AssignVariableOp_52AssignVariableOp*assignvariableop_52_nadam_hidden1_kernel_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53°
AssignVariableOp_53AssignVariableOp(assignvariableop_53_nadam_hidden1_bias_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54²
AssignVariableOp_54AssignVariableOp*assignvariableop_54_nadam_hidden2_kernel_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55°
AssignVariableOp_55AssignVariableOp(assignvariableop_55_nadam_hidden2_bias_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56²
AssignVariableOp_56AssignVariableOp*assignvariableop_56_nadam_hidden3_kernel_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57°
AssignVariableOp_57AssignVariableOp(assignvariableop_57_nadam_hidden3_bias_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58²
AssignVariableOp_58AssignVariableOp*assignvariableop_58_nadam_network_kernel_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59°
AssignVariableOp_59AssignVariableOp(assignvariableop_59_nadam_network_bias_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_599
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp
Identity_60Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_60ù

Identity_61IdentityIdentity_60:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_61"#
identity_61Identity_61:output:0*
_input_shapes|
z: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
¬	

C__inference_FleetEmb_layer_call_and_return_conditional_losses_18644

inputs(
embedding_lookup_18638:
identity¢embedding_lookupù
embedding_lookupResourceGatherembedding_lookup_18638inputs",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*)
_class
loc:@embedding_lookup/18638*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02
embedding_lookupì
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@embedding_lookup/18638*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_lookup/Identity 
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¬	

C__inference_FleetEmb_layer_call_and_return_conditional_losses_17396

inputs(
embedding_lookup_17390:
identity¢embedding_lookupù
embedding_lookupResourceGatherembedding_lookup_17390inputs",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*)
_class
loc:@embedding_lookup/17390*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02
embedding_lookupì
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@embedding_lookup/17390*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_lookup/Identity 
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
 

ó
B__inference_hidden1_layer_call_and_return_conditional_losses_17534

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Tanh
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ò
y
%__inference_PcEmb_layer_call_fn_18667

inputs
unknown:P
identity¢StatefulPartitionedCallç
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_PcEmb_layer_call_and_return_conditional_losses_173832
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
«	

B__inference_FuelEmb_layer_call_and_return_conditional_losses_17422

inputs(
embedding_lookup_17416:
identity¢embedding_lookupù
embedding_lookupResourceGatherembedding_lookup_17416inputs",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*)
_class
loc:@embedding_lookup/17416*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02
embedding_lookupì
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@embedding_lookup/17416*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_lookup/Identity 
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Æ
F
*__inference_Usage_flat_layer_call_fn_18711

inputs
identityÃ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_Usage_flat_layer_call_and_return_conditional_losses_174822
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Â\


B__inference_model_3_layer_call_and_return_conditional_losses_18110

design
coverage
sex
fuel	
usage	
fleet

postalcode

loggam
pcemb_18049:P 
fleetemb_18052: 
usageemb_18055:
fuelemb_18058:
sexemb_18061:
covemb_18064:)
batch_normalization_4_18074:)
batch_normalization_4_18076:)
batch_normalization_4_18078:)
batch_normalization_4_18080:
hidden1_18083:
hidden1_18085:
hidden2_18088:

hidden2_18090:

hidden3_18093:

hidden3_18095:
network_18098:
network_18100: 
response_18104:
response_18106:
identity¢CovEmb/StatefulPartitionedCall¢ FleetEmb/StatefulPartitionedCall¢FuelEmb/StatefulPartitionedCall¢Network/StatefulPartitionedCall¢PcEmb/StatefulPartitionedCall¢ Response/StatefulPartitionedCall¢SexEmb/StatefulPartitionedCall¢ UsageEmb/StatefulPartitionedCall¢-batch_normalization_4/StatefulPartitionedCall¢hidden1/StatefulPartitionedCall¢hidden2/StatefulPartitionedCall¢hidden3/StatefulPartitionedCallû
PcEmb/StatefulPartitionedCallStatefulPartitionedCall
postalcodepcemb_18049*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_PcEmb_layer_call_and_return_conditional_losses_173832
PcEmb/StatefulPartitionedCall
 FleetEmb/StatefulPartitionedCallStatefulPartitionedCallfleetfleetemb_18052*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_FleetEmb_layer_call_and_return_conditional_losses_173962"
 FleetEmb/StatefulPartitionedCall
 UsageEmb/StatefulPartitionedCallStatefulPartitionedCallusageusageemb_18055*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_UsageEmb_layer_call_and_return_conditional_losses_174092"
 UsageEmb/StatefulPartitionedCallý
FuelEmb/StatefulPartitionedCallStatefulPartitionedCallfuelfuelemb_18058*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_FuelEmb_layer_call_and_return_conditional_losses_174222!
FuelEmb/StatefulPartitionedCallø
SexEmb/StatefulPartitionedCallStatefulPartitionedCallsexsexemb_18061*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_SexEmb_layer_call_and_return_conditional_losses_174352 
SexEmb/StatefulPartitionedCallý
CovEmb/StatefulPartitionedCallStatefulPartitionedCallcoveragecovemb_18064*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_CovEmb_layer_call_and_return_conditional_losses_174482 
CovEmb/StatefulPartitionedCallô
Cov_flat/PartitionedCallPartitionedCall'CovEmb/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_Cov_flat_layer_call_and_return_conditional_losses_174582
Cov_flat/PartitionedCallô
Sex_flat/PartitionedCallPartitionedCall'SexEmb/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_Sex_flat_layer_call_and_return_conditional_losses_174662
Sex_flat/PartitionedCallø
Fuel_flat/PartitionedCallPartitionedCall(FuelEmb/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_Fuel_flat_layer_call_and_return_conditional_losses_174742
Fuel_flat/PartitionedCallü
Usage_flat/PartitionedCallPartitionedCall)UsageEmb/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_Usage_flat_layer_call_and_return_conditional_losses_174822
Usage_flat/PartitionedCallü
Fleet_flat/PartitionedCallPartitionedCall)FleetEmb/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_Fleet_flat_layer_call_and_return_conditional_losses_174902
Fleet_flat/PartitionedCallð
Pc_flat/PartitionedCallPartitionedCall&PcEmb/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_Pc_flat_layer_call_and_return_conditional_losses_174982
Pc_flat/PartitionedCall¬
concate/PartitionedCallPartitionedCalldesign!Cov_flat/PartitionedCall:output:0!Sex_flat/PartitionedCall:output:0"Fuel_flat/PartitionedCall:output:0#Usage_flat/PartitionedCall:output:0#Fleet_flat/PartitionedCall:output:0 Pc_flat/PartitionedCall:output:0*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_concate_layer_call_and_return_conditional_losses_175122
concate/PartitionedCallª
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall concate/PartitionedCall:output:0batch_normalization_4_18074batch_normalization_4_18076batch_normalization_4_18078batch_normalization_4_18080*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_172152/
-batch_normalization_4/StatefulPartitionedCall¼
hidden1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0hidden1_18083hidden1_18085*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_hidden1_layer_call_and_return_conditional_losses_175342!
hidden1/StatefulPartitionedCall®
hidden2/StatefulPartitionedCallStatefulPartitionedCall(hidden1/StatefulPartitionedCall:output:0hidden2_18088hidden2_18090*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_hidden2_layer_call_and_return_conditional_losses_175512!
hidden2/StatefulPartitionedCall®
hidden3/StatefulPartitionedCallStatefulPartitionedCall(hidden2/StatefulPartitionedCall:output:0hidden3_18093hidden3_18095*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_hidden3_layer_call_and_return_conditional_losses_175682!
hidden3/StatefulPartitionedCall®
Network/StatefulPartitionedCallStatefulPartitionedCall(hidden3/StatefulPartitionedCall:output:0network_18098network_18100*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_Network_layer_call_and_return_conditional_losses_175842!
Network/StatefulPartitionedCallï
Add/PartitionedCallPartitionedCall(Network/StatefulPartitionedCall:output:0loggam*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_Add_layer_call_and_return_conditional_losses_175962
Add/PartitionedCall§
 Response/StatefulPartitionedCallStatefulPartitionedCallAdd/PartitionedCall:output:0response_18104response_18106*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_Response_layer_call_and_return_conditional_losses_176092"
 Response/StatefulPartitionedCall¢
IdentityIdentity)Response/StatefulPartitionedCall:output:0^CovEmb/StatefulPartitionedCall!^FleetEmb/StatefulPartitionedCall ^FuelEmb/StatefulPartitionedCall ^Network/StatefulPartitionedCall^PcEmb/StatefulPartitionedCall!^Response/StatefulPartitionedCall^SexEmb/StatefulPartitionedCall!^UsageEmb/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall ^hidden1/StatefulPartitionedCall ^hidden2/StatefulPartitionedCall ^hidden3/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Õ
_input_shapesÃ
À:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : 2@
CovEmb/StatefulPartitionedCallCovEmb/StatefulPartitionedCall2D
 FleetEmb/StatefulPartitionedCall FleetEmb/StatefulPartitionedCall2B
FuelEmb/StatefulPartitionedCallFuelEmb/StatefulPartitionedCall2B
Network/StatefulPartitionedCallNetwork/StatefulPartitionedCall2>
PcEmb/StatefulPartitionedCallPcEmb/StatefulPartitionedCall2D
 Response/StatefulPartitionedCall Response/StatefulPartitionedCall2@
SexEmb/StatefulPartitionedCallSexEmb/StatefulPartitionedCall2D
 UsageEmb/StatefulPartitionedCall UsageEmb/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2B
hidden1/StatefulPartitionedCallhidden1/StatefulPartitionedCall2B
hidden2/StatefulPartitionedCallhidden2/StatefulPartitionedCall2B
hidden3/StatefulPartitionedCallhidden3/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameDesign:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
Coverage:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameSex:MI
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameFuel:NJ
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameUsage:NJ
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameFleet:SO
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
PostalCode:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameLogGAM


ô
C__inference_Response_layer_call_and_return_conditional_losses_17609

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddU
ExpExpBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Exp
IdentityIdentityExp:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
£
ë
'__inference_model_3_layer_call_fn_18519
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
unknown:P
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:


unknown_12:


unknown_13:


unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:
identity¢StatefulPartitionedCall´
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18*'
Tin 
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_model_3_layer_call_and_return_conditional_losses_176162
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Õ
_input_shapesÃ
À:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/2:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/3:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/4:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/5:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/6:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/7
¬	

C__inference_UsageEmb_layer_call_and_return_conditional_losses_17409

inputs(
embedding_lookup_17403:
identity¢embedding_lookupù
embedding_lookupResourceGatherembedding_lookup_17403inputs",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*)
_class
loc:@embedding_lookup/17403*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02
embedding_lookupì
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@embedding_lookup/17403*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_lookup/Identity 
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ô
z
&__inference_CovEmb_layer_call_fn_18587

inputs
unknown:
identity¢StatefulPartitionedCallè
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_CovEmb_layer_call_and_return_conditional_losses_174482
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ª	

A__inference_CovEmb_layer_call_and_return_conditional_losses_17448

inputs(
embedding_lookup_17442:
identity¢embedding_lookupù
embedding_lookupResourceGatherembedding_lookup_17442inputs",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*)
_class
loc:@embedding_lookup/17442*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02
embedding_lookupì
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@embedding_lookup/17442*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_lookup/Identity 
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Û

´
B__inference_concate_layer_call_and_return_conditional_losses_18745
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis³
concatConcatV2inputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/2:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/3:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/4:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/5:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/6
Ú
^
B__inference_Pc_flat_layer_call_and_return_conditional_losses_18728

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Â
D
(__inference_Cov_flat_layer_call_fn_18678

inputs
identityÁ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_Cov_flat_layer_call_and_return_conditional_losses_174582
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ü
`
D__inference_Fuel_flat_layer_call_and_return_conditional_losses_17474

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ý
a
E__inference_Fleet_flat_layer_call_and_return_conditional_losses_17490

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ª	

A__inference_SexEmb_layer_call_and_return_conditional_losses_17435

inputs(
embedding_lookup_17429:
identity¢embedding_lookupù
embedding_lookupResourceGatherembedding_lookup_17429inputs",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*)
_class
loc:@embedding_lookup/17429*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02
embedding_lookupì
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@embedding_lookup/17429*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_lookup/Identity 
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ö
{
'__inference_FuelEmb_layer_call_fn_18619

inputs
unknown:
identity¢StatefulPartitionedCallé
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_FuelEmb_layer_call_and_return_conditional_losses_174222
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ý
Ð
5__inference_batch_normalization_4_layer_call_fn_18836

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_172752
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ø
|
(__inference_FleetEmb_layer_call_fn_18651

inputs
unknown:
identity¢StatefulPartitionedCallê
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_FleetEmb_layer_call_and_return_conditional_losses_173962
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

¯
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_18776

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1Û
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
À\


B__inference_model_3_layer_call_and_return_conditional_losses_18181

design
coverage
sex
fuel	
usage	
fleet

postalcode

loggam
pcemb_18120:P 
fleetemb_18123: 
usageemb_18126:
fuelemb_18129:
sexemb_18132:
covemb_18135:)
batch_normalization_4_18145:)
batch_normalization_4_18147:)
batch_normalization_4_18149:)
batch_normalization_4_18151:
hidden1_18154:
hidden1_18156:
hidden2_18159:

hidden2_18161:

hidden3_18164:

hidden3_18166:
network_18169:
network_18171: 
response_18175:
response_18177:
identity¢CovEmb/StatefulPartitionedCall¢ FleetEmb/StatefulPartitionedCall¢FuelEmb/StatefulPartitionedCall¢Network/StatefulPartitionedCall¢PcEmb/StatefulPartitionedCall¢ Response/StatefulPartitionedCall¢SexEmb/StatefulPartitionedCall¢ UsageEmb/StatefulPartitionedCall¢-batch_normalization_4/StatefulPartitionedCall¢hidden1/StatefulPartitionedCall¢hidden2/StatefulPartitionedCall¢hidden3/StatefulPartitionedCallû
PcEmb/StatefulPartitionedCallStatefulPartitionedCall
postalcodepcemb_18120*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_PcEmb_layer_call_and_return_conditional_losses_173832
PcEmb/StatefulPartitionedCall
 FleetEmb/StatefulPartitionedCallStatefulPartitionedCallfleetfleetemb_18123*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_FleetEmb_layer_call_and_return_conditional_losses_173962"
 FleetEmb/StatefulPartitionedCall
 UsageEmb/StatefulPartitionedCallStatefulPartitionedCallusageusageemb_18126*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_UsageEmb_layer_call_and_return_conditional_losses_174092"
 UsageEmb/StatefulPartitionedCallý
FuelEmb/StatefulPartitionedCallStatefulPartitionedCallfuelfuelemb_18129*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_FuelEmb_layer_call_and_return_conditional_losses_174222!
FuelEmb/StatefulPartitionedCallø
SexEmb/StatefulPartitionedCallStatefulPartitionedCallsexsexemb_18132*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_SexEmb_layer_call_and_return_conditional_losses_174352 
SexEmb/StatefulPartitionedCallý
CovEmb/StatefulPartitionedCallStatefulPartitionedCallcoveragecovemb_18135*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_CovEmb_layer_call_and_return_conditional_losses_174482 
CovEmb/StatefulPartitionedCallô
Cov_flat/PartitionedCallPartitionedCall'CovEmb/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_Cov_flat_layer_call_and_return_conditional_losses_174582
Cov_flat/PartitionedCallô
Sex_flat/PartitionedCallPartitionedCall'SexEmb/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_Sex_flat_layer_call_and_return_conditional_losses_174662
Sex_flat/PartitionedCallø
Fuel_flat/PartitionedCallPartitionedCall(FuelEmb/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_Fuel_flat_layer_call_and_return_conditional_losses_174742
Fuel_flat/PartitionedCallü
Usage_flat/PartitionedCallPartitionedCall)UsageEmb/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_Usage_flat_layer_call_and_return_conditional_losses_174822
Usage_flat/PartitionedCallü
Fleet_flat/PartitionedCallPartitionedCall)FleetEmb/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_Fleet_flat_layer_call_and_return_conditional_losses_174902
Fleet_flat/PartitionedCallð
Pc_flat/PartitionedCallPartitionedCall&PcEmb/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_Pc_flat_layer_call_and_return_conditional_losses_174982
Pc_flat/PartitionedCall¬
concate/PartitionedCallPartitionedCalldesign!Cov_flat/PartitionedCall:output:0!Sex_flat/PartitionedCall:output:0"Fuel_flat/PartitionedCall:output:0#Usage_flat/PartitionedCall:output:0#Fleet_flat/PartitionedCall:output:0 Pc_flat/PartitionedCall:output:0*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_concate_layer_call_and_return_conditional_losses_175122
concate/PartitionedCall¨
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall concate/PartitionedCall:output:0batch_normalization_4_18145batch_normalization_4_18147batch_normalization_4_18149batch_normalization_4_18151*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_172752/
-batch_normalization_4/StatefulPartitionedCall¼
hidden1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0hidden1_18154hidden1_18156*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_hidden1_layer_call_and_return_conditional_losses_175342!
hidden1/StatefulPartitionedCall®
hidden2/StatefulPartitionedCallStatefulPartitionedCall(hidden1/StatefulPartitionedCall:output:0hidden2_18159hidden2_18161*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_hidden2_layer_call_and_return_conditional_losses_175512!
hidden2/StatefulPartitionedCall®
hidden3/StatefulPartitionedCallStatefulPartitionedCall(hidden2/StatefulPartitionedCall:output:0hidden3_18164hidden3_18166*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_hidden3_layer_call_and_return_conditional_losses_175682!
hidden3/StatefulPartitionedCall®
Network/StatefulPartitionedCallStatefulPartitionedCall(hidden3/StatefulPartitionedCall:output:0network_18169network_18171*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_Network_layer_call_and_return_conditional_losses_175842!
Network/StatefulPartitionedCallï
Add/PartitionedCallPartitionedCall(Network/StatefulPartitionedCall:output:0loggam*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_Add_layer_call_and_return_conditional_losses_175962
Add/PartitionedCall§
 Response/StatefulPartitionedCallStatefulPartitionedCallAdd/PartitionedCall:output:0response_18175response_18177*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_Response_layer_call_and_return_conditional_losses_176092"
 Response/StatefulPartitionedCall¢
IdentityIdentity)Response/StatefulPartitionedCall:output:0^CovEmb/StatefulPartitionedCall!^FleetEmb/StatefulPartitionedCall ^FuelEmb/StatefulPartitionedCall ^Network/StatefulPartitionedCall^PcEmb/StatefulPartitionedCall!^Response/StatefulPartitionedCall^SexEmb/StatefulPartitionedCall!^UsageEmb/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall ^hidden1/StatefulPartitionedCall ^hidden2/StatefulPartitionedCall ^hidden3/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Õ
_input_shapesÃ
À:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : 2@
CovEmb/StatefulPartitionedCallCovEmb/StatefulPartitionedCall2D
 FleetEmb/StatefulPartitionedCall FleetEmb/StatefulPartitionedCall2B
FuelEmb/StatefulPartitionedCallFuelEmb/StatefulPartitionedCall2B
Network/StatefulPartitionedCallNetwork/StatefulPartitionedCall2>
PcEmb/StatefulPartitionedCallPcEmb/StatefulPartitionedCall2D
 Response/StatefulPartitionedCall Response/StatefulPartitionedCall2@
SexEmb/StatefulPartitionedCallSexEmb/StatefulPartitionedCall2D
 UsageEmb/StatefulPartitionedCall UsageEmb/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2B
hidden1/StatefulPartitionedCallhidden1/StatefulPartitionedCall2B
hidden2/StatefulPartitionedCallhidden2/StatefulPartitionedCall2B
hidden3/StatefulPartitionedCallhidden3/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameDesign:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
Coverage:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameSex:MI
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameFuel:NJ
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameUsage:NJ
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameFleet:SO
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
PostalCode:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameLogGAM


ô
C__inference_Response_layer_call_and_return_conditional_losses_18938

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddU
ExpExpBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Exp
IdentityIdentityExp:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ñ
ý
B__inference_model_3_layer_call_and_return_conditional_losses_18347
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7.
pcemb_embedding_lookup_18251:P1
fleetemb_embedding_lookup_18256:1
usageemb_embedding_lookup_18261:0
fuelemb_embedding_lookup_18266:/
sexemb_embedding_lookup_18271:/
covemb_embedding_lookup_18276:E
7batch_normalization_4_batchnorm_readvariableop_resource:I
;batch_normalization_4_batchnorm_mul_readvariableop_resource:G
9batch_normalization_4_batchnorm_readvariableop_1_resource:G
9batch_normalization_4_batchnorm_readvariableop_2_resource:8
&hidden1_matmul_readvariableop_resource:5
'hidden1_biasadd_readvariableop_resource:8
&hidden2_matmul_readvariableop_resource:
5
'hidden2_biasadd_readvariableop_resource:
8
&hidden3_matmul_readvariableop_resource:
5
'hidden3_biasadd_readvariableop_resource:8
&network_matmul_readvariableop_resource:5
'network_biasadd_readvariableop_resource:9
'response_matmul_readvariableop_resource:6
(response_biasadd_readvariableop_resource:
identity¢CovEmb/embedding_lookup¢FleetEmb/embedding_lookup¢FuelEmb/embedding_lookup¢Network/BiasAdd/ReadVariableOp¢Network/MatMul/ReadVariableOp¢PcEmb/embedding_lookup¢Response/BiasAdd/ReadVariableOp¢Response/MatMul/ReadVariableOp¢SexEmb/embedding_lookup¢UsageEmb/embedding_lookup¢.batch_normalization_4/batchnorm/ReadVariableOp¢0batch_normalization_4/batchnorm/ReadVariableOp_1¢0batch_normalization_4/batchnorm/ReadVariableOp_2¢2batch_normalization_4/batchnorm/mul/ReadVariableOp¢hidden1/BiasAdd/ReadVariableOp¢hidden1/MatMul/ReadVariableOp¢hidden2/BiasAdd/ReadVariableOp¢hidden2/MatMul/ReadVariableOp¢hidden3/BiasAdd/ReadVariableOp¢hidden3/MatMul/ReadVariableOp
PcEmb/embedding_lookupResourceGatherpcemb_embedding_lookup_18251inputs_6",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*/
_class%
#!loc:@PcEmb/embedding_lookup/18251*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02
PcEmb/embedding_lookup
PcEmb/embedding_lookup/IdentityIdentityPcEmb/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*/
_class%
#!loc:@PcEmb/embedding_lookup/18251*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
PcEmb/embedding_lookup/Identity²
!PcEmb/embedding_lookup/Identity_1Identity(PcEmb/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!PcEmb/embedding_lookup/Identity_1
FleetEmb/embedding_lookupResourceGatherfleetemb_embedding_lookup_18256inputs_5",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*2
_class(
&$loc:@FleetEmb/embedding_lookup/18256*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02
FleetEmb/embedding_lookup
"FleetEmb/embedding_lookup/IdentityIdentity"FleetEmb/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*2
_class(
&$loc:@FleetEmb/embedding_lookup/18256*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"FleetEmb/embedding_lookup/Identity»
$FleetEmb/embedding_lookup/Identity_1Identity+FleetEmb/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$FleetEmb/embedding_lookup/Identity_1
UsageEmb/embedding_lookupResourceGatherusageemb_embedding_lookup_18261inputs_4",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*2
_class(
&$loc:@UsageEmb/embedding_lookup/18261*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02
UsageEmb/embedding_lookup
"UsageEmb/embedding_lookup/IdentityIdentity"UsageEmb/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*2
_class(
&$loc:@UsageEmb/embedding_lookup/18261*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"UsageEmb/embedding_lookup/Identity»
$UsageEmb/embedding_lookup/Identity_1Identity+UsageEmb/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$UsageEmb/embedding_lookup/Identity_1
FuelEmb/embedding_lookupResourceGatherfuelemb_embedding_lookup_18266inputs_3",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*1
_class'
%#loc:@FuelEmb/embedding_lookup/18266*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02
FuelEmb/embedding_lookup
!FuelEmb/embedding_lookup/IdentityIdentity!FuelEmb/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*1
_class'
%#loc:@FuelEmb/embedding_lookup/18266*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!FuelEmb/embedding_lookup/Identity¸
#FuelEmb/embedding_lookup/Identity_1Identity*FuelEmb/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#FuelEmb/embedding_lookup/Identity_1
SexEmb/embedding_lookupResourceGathersexemb_embedding_lookup_18271inputs_2",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*0
_class&
$"loc:@SexEmb/embedding_lookup/18271*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02
SexEmb/embedding_lookup
 SexEmb/embedding_lookup/IdentityIdentity SexEmb/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*0
_class&
$"loc:@SexEmb/embedding_lookup/18271*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 SexEmb/embedding_lookup/Identityµ
"SexEmb/embedding_lookup/Identity_1Identity)SexEmb/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"SexEmb/embedding_lookup/Identity_1
CovEmb/embedding_lookupResourceGathercovemb_embedding_lookup_18276inputs_1",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*0
_class&
$"loc:@CovEmb/embedding_lookup/18276*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02
CovEmb/embedding_lookup
 CovEmb/embedding_lookup/IdentityIdentity CovEmb/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*0
_class&
$"loc:@CovEmb/embedding_lookup/18276*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 CovEmb/embedding_lookup/Identityµ
"CovEmb/embedding_lookup/Identity_1Identity)CovEmb/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"CovEmb/embedding_lookup/Identity_1q
Cov_flat/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
Cov_flat/Const§
Cov_flat/ReshapeReshape+CovEmb/embedding_lookup/Identity_1:output:0Cov_flat/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Cov_flat/Reshapeq
Sex_flat/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
Sex_flat/Const§
Sex_flat/ReshapeReshape+SexEmb/embedding_lookup/Identity_1:output:0Sex_flat/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Sex_flat/Reshapes
Fuel_flat/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
Fuel_flat/Const«
Fuel_flat/ReshapeReshape,FuelEmb/embedding_lookup/Identity_1:output:0Fuel_flat/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Fuel_flat/Reshapeu
Usage_flat/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
Usage_flat/Const¯
Usage_flat/ReshapeReshape-UsageEmb/embedding_lookup/Identity_1:output:0Usage_flat/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Usage_flat/Reshapeu
Fleet_flat/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
Fleet_flat/Const¯
Fleet_flat/ReshapeReshape-FleetEmb/embedding_lookup/Identity_1:output:0Fleet_flat/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Fleet_flat/Reshapeo
Pc_flat/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
Pc_flat/Const£
Pc_flat/ReshapeReshape*PcEmb/embedding_lookup/Identity_1:output:0Pc_flat/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Pc_flat/Reshapel
concate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concate/concat/axisµ
concate/concatConcatV2inputs_0Cov_flat/Reshape:output:0Sex_flat/Reshape:output:0Fuel_flat/Reshape:output:0Usage_flat/Reshape:output:0Fleet_flat/Reshape:output:0Pc_flat/Reshape:output:0concate/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
concate/concatÔ
.batch_normalization_4/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_4_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype020
.batch_normalization_4/batchnorm/ReadVariableOp
%batch_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2'
%batch_normalization_4/batchnorm/add/yà
#batch_normalization_4/batchnorm/addAddV26batch_normalization_4/batchnorm/ReadVariableOp:value:0.batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes
:2%
#batch_normalization_4/batchnorm/add¥
%batch_normalization_4/batchnorm/RsqrtRsqrt'batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes
:2'
%batch_normalization_4/batchnorm/Rsqrtà
2batch_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype024
2batch_normalization_4/batchnorm/mul/ReadVariableOpÝ
#batch_normalization_4/batchnorm/mulMul)batch_normalization_4/batchnorm/Rsqrt:y:0:batch_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2%
#batch_normalization_4/batchnorm/mulÉ
%batch_normalization_4/batchnorm/mul_1Mulconcate/concat:output:0'batch_normalization_4/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%batch_normalization_4/batchnorm/mul_1Ú
0batch_normalization_4/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_4_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype022
0batch_normalization_4/batchnorm/ReadVariableOp_1Ý
%batch_normalization_4/batchnorm/mul_2Mul8batch_normalization_4/batchnorm/ReadVariableOp_1:value:0'batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes
:2'
%batch_normalization_4/batchnorm/mul_2Ú
0batch_normalization_4/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_4_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype022
0batch_normalization_4/batchnorm/ReadVariableOp_2Û
#batch_normalization_4/batchnorm/subSub8batch_normalization_4/batchnorm/ReadVariableOp_2:value:0)batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2%
#batch_normalization_4/batchnorm/subÝ
%batch_normalization_4/batchnorm/add_1AddV2)batch_normalization_4/batchnorm/mul_1:z:0'batch_normalization_4/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%batch_normalization_4/batchnorm/add_1¥
hidden1/MatMul/ReadVariableOpReadVariableOp&hidden1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
hidden1/MatMul/ReadVariableOp®
hidden1/MatMulMatMul)batch_normalization_4/batchnorm/add_1:z:0%hidden1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
hidden1/MatMul¤
hidden1/BiasAdd/ReadVariableOpReadVariableOp'hidden1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
hidden1/BiasAdd/ReadVariableOp¡
hidden1/BiasAddBiasAddhidden1/MatMul:product:0&hidden1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
hidden1/BiasAddp
hidden1/TanhTanhhidden1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
hidden1/Tanh¥
hidden2/MatMul/ReadVariableOpReadVariableOp&hidden2_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02
hidden2/MatMul/ReadVariableOp
hidden2/MatMulMatMulhidden1/Tanh:y:0%hidden2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
hidden2/MatMul¤
hidden2/BiasAdd/ReadVariableOpReadVariableOp'hidden2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02 
hidden2/BiasAdd/ReadVariableOp¡
hidden2/BiasAddBiasAddhidden2/MatMul:product:0&hidden2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
hidden2/BiasAddp
hidden2/TanhTanhhidden2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
hidden2/Tanh¥
hidden3/MatMul/ReadVariableOpReadVariableOp&hidden3_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02
hidden3/MatMul/ReadVariableOp
hidden3/MatMulMatMulhidden2/Tanh:y:0%hidden3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
hidden3/MatMul¤
hidden3/BiasAdd/ReadVariableOpReadVariableOp'hidden3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
hidden3/BiasAdd/ReadVariableOp¡
hidden3/BiasAddBiasAddhidden3/MatMul:product:0&hidden3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
hidden3/BiasAddp
hidden3/TanhTanhhidden3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
hidden3/Tanh¥
Network/MatMul/ReadVariableOpReadVariableOp&network_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
Network/MatMul/ReadVariableOp
Network/MatMulMatMulhidden3/Tanh:y:0%Network/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Network/MatMul¤
Network/BiasAdd/ReadVariableOpReadVariableOp'network_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
Network/BiasAdd/ReadVariableOp¡
Network/BiasAddBiasAddNetwork/MatMul:product:0&Network/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Network/BiasAddq
Add/addAddV2Network/BiasAdd:output:0inputs_7*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Add/add¨
Response/MatMul/ReadVariableOpReadVariableOp'response_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
Response/MatMul/ReadVariableOp
Response/MatMulMatMulAdd/add:z:0&Response/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Response/MatMul§
Response/BiasAdd/ReadVariableOpReadVariableOp(response_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
Response/BiasAdd/ReadVariableOp¥
Response/BiasAddBiasAddResponse/MatMul:product:0'Response/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Response/BiasAddp
Response/ExpExpResponse/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Response/Exp
IdentityIdentityResponse/Exp:y:0^CovEmb/embedding_lookup^FleetEmb/embedding_lookup^FuelEmb/embedding_lookup^Network/BiasAdd/ReadVariableOp^Network/MatMul/ReadVariableOp^PcEmb/embedding_lookup ^Response/BiasAdd/ReadVariableOp^Response/MatMul/ReadVariableOp^SexEmb/embedding_lookup^UsageEmb/embedding_lookup/^batch_normalization_4/batchnorm/ReadVariableOp1^batch_normalization_4/batchnorm/ReadVariableOp_11^batch_normalization_4/batchnorm/ReadVariableOp_23^batch_normalization_4/batchnorm/mul/ReadVariableOp^hidden1/BiasAdd/ReadVariableOp^hidden1/MatMul/ReadVariableOp^hidden2/BiasAdd/ReadVariableOp^hidden2/MatMul/ReadVariableOp^hidden3/BiasAdd/ReadVariableOp^hidden3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Õ
_input_shapesÃ
À:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : 22
CovEmb/embedding_lookupCovEmb/embedding_lookup26
FleetEmb/embedding_lookupFleetEmb/embedding_lookup24
FuelEmb/embedding_lookupFuelEmb/embedding_lookup2@
Network/BiasAdd/ReadVariableOpNetwork/BiasAdd/ReadVariableOp2>
Network/MatMul/ReadVariableOpNetwork/MatMul/ReadVariableOp20
PcEmb/embedding_lookupPcEmb/embedding_lookup2B
Response/BiasAdd/ReadVariableOpResponse/BiasAdd/ReadVariableOp2@
Response/MatMul/ReadVariableOpResponse/MatMul/ReadVariableOp22
SexEmb/embedding_lookupSexEmb/embedding_lookup26
UsageEmb/embedding_lookupUsageEmb/embedding_lookup2`
.batch_normalization_4/batchnorm/ReadVariableOp.batch_normalization_4/batchnorm/ReadVariableOp2d
0batch_normalization_4/batchnorm/ReadVariableOp_10batch_normalization_4/batchnorm/ReadVariableOp_12d
0batch_normalization_4/batchnorm/ReadVariableOp_20batch_normalization_4/batchnorm/ReadVariableOp_22h
2batch_normalization_4/batchnorm/mul/ReadVariableOp2batch_normalization_4/batchnorm/mul/ReadVariableOp2@
hidden1/BiasAdd/ReadVariableOphidden1/BiasAdd/ReadVariableOp2>
hidden1/MatMul/ReadVariableOphidden1/MatMul/ReadVariableOp2@
hidden2/BiasAdd/ReadVariableOphidden2/BiasAdd/ReadVariableOp2>
hidden2/MatMul/ReadVariableOphidden2/MatMul/ReadVariableOp2@
hidden3/BiasAdd/ReadVariableOphidden3/BiasAdd/ReadVariableOp2>
hidden3/MatMul/ReadVariableOphidden3/MatMul/ReadVariableOp:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/2:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/3:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/4:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/5:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/6:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/7
Û
_
C__inference_Cov_flat_layer_call_and_return_conditional_losses_17458

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ß\


B__inference_model_3_layer_call_and_return_conditional_losses_17944

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
pcemb_17883:P 
fleetemb_17886: 
usageemb_17889:
fuelemb_17892:
sexemb_17895:
covemb_17898:)
batch_normalization_4_17908:)
batch_normalization_4_17910:)
batch_normalization_4_17912:)
batch_normalization_4_17914:
hidden1_17917:
hidden1_17919:
hidden2_17922:

hidden2_17924:

hidden3_17927:

hidden3_17929:
network_17932:
network_17934: 
response_17938:
response_17940:
identity¢CovEmb/StatefulPartitionedCall¢ FleetEmb/StatefulPartitionedCall¢FuelEmb/StatefulPartitionedCall¢Network/StatefulPartitionedCall¢PcEmb/StatefulPartitionedCall¢ Response/StatefulPartitionedCall¢SexEmb/StatefulPartitionedCall¢ UsageEmb/StatefulPartitionedCall¢-batch_normalization_4/StatefulPartitionedCall¢hidden1/StatefulPartitionedCall¢hidden2/StatefulPartitionedCall¢hidden3/StatefulPartitionedCallù
PcEmb/StatefulPartitionedCallStatefulPartitionedCallinputs_6pcemb_17883*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_PcEmb_layer_call_and_return_conditional_losses_173832
PcEmb/StatefulPartitionedCall
 FleetEmb/StatefulPartitionedCallStatefulPartitionedCallinputs_5fleetemb_17886*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_FleetEmb_layer_call_and_return_conditional_losses_173962"
 FleetEmb/StatefulPartitionedCall
 UsageEmb/StatefulPartitionedCallStatefulPartitionedCallinputs_4usageemb_17889*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_UsageEmb_layer_call_and_return_conditional_losses_174092"
 UsageEmb/StatefulPartitionedCall
FuelEmb/StatefulPartitionedCallStatefulPartitionedCallinputs_3fuelemb_17892*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_FuelEmb_layer_call_and_return_conditional_losses_174222!
FuelEmb/StatefulPartitionedCallý
SexEmb/StatefulPartitionedCallStatefulPartitionedCallinputs_2sexemb_17895*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_SexEmb_layer_call_and_return_conditional_losses_174352 
SexEmb/StatefulPartitionedCallý
CovEmb/StatefulPartitionedCallStatefulPartitionedCallinputs_1covemb_17898*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_CovEmb_layer_call_and_return_conditional_losses_174482 
CovEmb/StatefulPartitionedCallô
Cov_flat/PartitionedCallPartitionedCall'CovEmb/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_Cov_flat_layer_call_and_return_conditional_losses_174582
Cov_flat/PartitionedCallô
Sex_flat/PartitionedCallPartitionedCall'SexEmb/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_Sex_flat_layer_call_and_return_conditional_losses_174662
Sex_flat/PartitionedCallø
Fuel_flat/PartitionedCallPartitionedCall(FuelEmb/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_Fuel_flat_layer_call_and_return_conditional_losses_174742
Fuel_flat/PartitionedCallü
Usage_flat/PartitionedCallPartitionedCall)UsageEmb/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_Usage_flat_layer_call_and_return_conditional_losses_174822
Usage_flat/PartitionedCallü
Fleet_flat/PartitionedCallPartitionedCall)FleetEmb/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_Fleet_flat_layer_call_and_return_conditional_losses_174902
Fleet_flat/PartitionedCallð
Pc_flat/PartitionedCallPartitionedCall&PcEmb/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_Pc_flat_layer_call_and_return_conditional_losses_174982
Pc_flat/PartitionedCall¬
concate/PartitionedCallPartitionedCallinputs!Cov_flat/PartitionedCall:output:0!Sex_flat/PartitionedCall:output:0"Fuel_flat/PartitionedCall:output:0#Usage_flat/PartitionedCall:output:0#Fleet_flat/PartitionedCall:output:0 Pc_flat/PartitionedCall:output:0*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_concate_layer_call_and_return_conditional_losses_175122
concate/PartitionedCall¨
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall concate/PartitionedCall:output:0batch_normalization_4_17908batch_normalization_4_17910batch_normalization_4_17912batch_normalization_4_17914*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_172752/
-batch_normalization_4/StatefulPartitionedCall¼
hidden1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0hidden1_17917hidden1_17919*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_hidden1_layer_call_and_return_conditional_losses_175342!
hidden1/StatefulPartitionedCall®
hidden2/StatefulPartitionedCallStatefulPartitionedCall(hidden1/StatefulPartitionedCall:output:0hidden2_17922hidden2_17924*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_hidden2_layer_call_and_return_conditional_losses_175512!
hidden2/StatefulPartitionedCall®
hidden3/StatefulPartitionedCallStatefulPartitionedCall(hidden2/StatefulPartitionedCall:output:0hidden3_17927hidden3_17929*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_hidden3_layer_call_and_return_conditional_losses_175682!
hidden3/StatefulPartitionedCall®
Network/StatefulPartitionedCallStatefulPartitionedCall(hidden3/StatefulPartitionedCall:output:0network_17932network_17934*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_Network_layer_call_and_return_conditional_losses_175842!
Network/StatefulPartitionedCallñ
Add/PartitionedCallPartitionedCall(Network/StatefulPartitionedCall:output:0inputs_7*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_Add_layer_call_and_return_conditional_losses_175962
Add/PartitionedCall§
 Response/StatefulPartitionedCallStatefulPartitionedCallAdd/PartitionedCall:output:0response_17938response_17940*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_Response_layer_call_and_return_conditional_losses_176092"
 Response/StatefulPartitionedCall¢
IdentityIdentity)Response/StatefulPartitionedCall:output:0^CovEmb/StatefulPartitionedCall!^FleetEmb/StatefulPartitionedCall ^FuelEmb/StatefulPartitionedCall ^Network/StatefulPartitionedCall^PcEmb/StatefulPartitionedCall!^Response/StatefulPartitionedCall^SexEmb/StatefulPartitionedCall!^UsageEmb/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall ^hidden1/StatefulPartitionedCall ^hidden2/StatefulPartitionedCall ^hidden3/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Õ
_input_shapesÃ
À:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : 2@
CovEmb/StatefulPartitionedCallCovEmb/StatefulPartitionedCall2D
 FleetEmb/StatefulPartitionedCall FleetEmb/StatefulPartitionedCall2B
FuelEmb/StatefulPartitionedCallFuelEmb/StatefulPartitionedCall2B
Network/StatefulPartitionedCallNetwork/StatefulPartitionedCall2>
PcEmb/StatefulPartitionedCallPcEmb/StatefulPartitionedCall2D
 Response/StatefulPartitionedCall Response/StatefulPartitionedCall2@
SexEmb/StatefulPartitionedCallSexEmb/StatefulPartitionedCall2D
 UsageEmb/StatefulPartitionedCall UsageEmb/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2B
hidden1/StatefulPartitionedCallhidden1/StatefulPartitionedCall2B
hidden2/StatefulPartitionedCallhidden2/StatefulPartitionedCall2B
hidden3/StatefulPartitionedCallhidden3/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¬	

C__inference_UsageEmb_layer_call_and_return_conditional_losses_18628

inputs(
embedding_lookup_18622:
identity¢embedding_lookupù
embedding_lookupResourceGatherembedding_lookup_18622inputs",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*)
_class
loc:@embedding_lookup/18622*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02
embedding_lookupì
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@embedding_lookup/18622*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_lookup/Identity 
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ð
Ú
'__inference_model_3_layer_call_fn_17659

design
coverage
sex
fuel	
usage	
fleet

postalcode

loggam
unknown:P
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:


unknown_12:


unknown_13:


unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:
identity¢StatefulPartitionedCall£
StatefulPartitionedCallStatefulPartitionedCalldesigncoveragesexfuelusagefleet
postalcodeloggamunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18*'
Tin 
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_model_3_layer_call_and_return_conditional_losses_176162
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Õ
_input_shapesÃ
À:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameDesign:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
Coverage:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameSex:MI
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameFuel:NJ
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameUsage:NJ
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameFleet:SO
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
PostalCode:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameLogGAM
î
Ú
'__inference_model_3_layer_call_fn_18039

design
coverage
sex
fuel	
usage	
fleet

postalcode

loggam
unknown:P
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:


unknown_12:


unknown_13:


unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:
identity¢StatefulPartitionedCall¡
StatefulPartitionedCallStatefulPartitionedCalldesigncoveragesexfuelusagefleet
postalcodeloggamunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18*'
Tin 
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_model_3_layer_call_and_return_conditional_losses_179442
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Õ
_input_shapesÃ
À:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameDesign:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
Coverage:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameSex:MI
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameFuel:NJ
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameUsage:NJ
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameFleet:SO
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
PostalCode:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameLogGAM
Ú
^
B__inference_Pc_flat_layer_call_and_return_conditional_losses_17498

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÿ
Ð
5__inference_batch_normalization_4_layer_call_fn_18823

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_172152
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
À
C
'__inference_Pc_flat_layer_call_fn_18733

inputs
identityÀ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_Pc_flat_layer_call_and_return_conditional_losses_174982
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Î	
ó
B__inference_Network_layer_call_and_return_conditional_losses_18906

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ý
a
E__inference_Fleet_flat_layer_call_and_return_conditional_losses_18717

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ô)
é
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_17275

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:2
moments/StopGradient¤
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices²
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay¤
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/sub
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/mul¿
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decayª
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp 
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/sub
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/mulÉ
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1
IdentityIdentitybatchnorm/add_1:z:0^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ª	

A__inference_CovEmb_layer_call_and_return_conditional_losses_18580

inputs(
embedding_lookup_18574:
identity¢embedding_lookupù
embedding_lookupResourceGatherembedding_lookup_18574inputs",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*)
_class
loc:@embedding_lookup/18574*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02
embedding_lookupì
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@embedding_lookup/18574*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_lookup/Identity 
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Û
_
C__inference_Sex_flat_layer_call_and_return_conditional_losses_17466

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ô
z
&__inference_SexEmb_layer_call_fn_18603

inputs
unknown:
identity¢StatefulPartitionedCallè
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_SexEmb_layer_call_and_return_conditional_losses_174352
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Æ
h
>__inference_Add_layer_call_and_return_conditional_losses_17596

inputs
inputs_1
identityW
addAddV2inputsinputs_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add[
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ª	

A__inference_SexEmb_layer_call_and_return_conditional_losses_18596

inputs(
embedding_lookup_18590:
identity¢embedding_lookupù
embedding_lookupResourceGatherembedding_lookup_18590inputs",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*)
_class
loc:@embedding_lookup/18590*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02
embedding_lookupì
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@embedding_lookup/18590*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_lookup/Identity 
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


'__inference_hidden3_layer_call_fn_18896

inputs
unknown:

	unknown_0:
identity¢StatefulPartitionedCallò
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_hidden3_layer_call_and_return_conditional_losses_175682
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
Æ
F
*__inference_Fleet_flat_layer_call_fn_18722

inputs
identityÃ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_Fleet_flat_layer_call_and_return_conditional_losses_174902
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
 

ó
B__inference_hidden2_layer_call_and_return_conditional_losses_18867

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:

identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
Tanh
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
 

ó
B__inference_hidden1_layer_call_and_return_conditional_losses_18847

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Tanh
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Û
_
C__inference_Cov_flat_layer_call_and_return_conditional_losses_18673

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ý
a
E__inference_Usage_flat_layer_call_and_return_conditional_losses_17482

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
á\


B__inference_model_3_layer_call_and_return_conditional_losses_17616

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
pcemb_17384:P 
fleetemb_17397: 
usageemb_17410:
fuelemb_17423:
sexemb_17436:
covemb_17449:)
batch_normalization_4_17514:)
batch_normalization_4_17516:)
batch_normalization_4_17518:)
batch_normalization_4_17520:
hidden1_17535:
hidden1_17537:
hidden2_17552:

hidden2_17554:

hidden3_17569:

hidden3_17571:
network_17585:
network_17587: 
response_17610:
response_17612:
identity¢CovEmb/StatefulPartitionedCall¢ FleetEmb/StatefulPartitionedCall¢FuelEmb/StatefulPartitionedCall¢Network/StatefulPartitionedCall¢PcEmb/StatefulPartitionedCall¢ Response/StatefulPartitionedCall¢SexEmb/StatefulPartitionedCall¢ UsageEmb/StatefulPartitionedCall¢-batch_normalization_4/StatefulPartitionedCall¢hidden1/StatefulPartitionedCall¢hidden2/StatefulPartitionedCall¢hidden3/StatefulPartitionedCallù
PcEmb/StatefulPartitionedCallStatefulPartitionedCallinputs_6pcemb_17384*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_PcEmb_layer_call_and_return_conditional_losses_173832
PcEmb/StatefulPartitionedCall
 FleetEmb/StatefulPartitionedCallStatefulPartitionedCallinputs_5fleetemb_17397*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_FleetEmb_layer_call_and_return_conditional_losses_173962"
 FleetEmb/StatefulPartitionedCall
 UsageEmb/StatefulPartitionedCallStatefulPartitionedCallinputs_4usageemb_17410*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_UsageEmb_layer_call_and_return_conditional_losses_174092"
 UsageEmb/StatefulPartitionedCall
FuelEmb/StatefulPartitionedCallStatefulPartitionedCallinputs_3fuelemb_17423*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_FuelEmb_layer_call_and_return_conditional_losses_174222!
FuelEmb/StatefulPartitionedCallý
SexEmb/StatefulPartitionedCallStatefulPartitionedCallinputs_2sexemb_17436*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_SexEmb_layer_call_and_return_conditional_losses_174352 
SexEmb/StatefulPartitionedCallý
CovEmb/StatefulPartitionedCallStatefulPartitionedCallinputs_1covemb_17449*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_CovEmb_layer_call_and_return_conditional_losses_174482 
CovEmb/StatefulPartitionedCallô
Cov_flat/PartitionedCallPartitionedCall'CovEmb/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_Cov_flat_layer_call_and_return_conditional_losses_174582
Cov_flat/PartitionedCallô
Sex_flat/PartitionedCallPartitionedCall'SexEmb/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_Sex_flat_layer_call_and_return_conditional_losses_174662
Sex_flat/PartitionedCallø
Fuel_flat/PartitionedCallPartitionedCall(FuelEmb/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_Fuel_flat_layer_call_and_return_conditional_losses_174742
Fuel_flat/PartitionedCallü
Usage_flat/PartitionedCallPartitionedCall)UsageEmb/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_Usage_flat_layer_call_and_return_conditional_losses_174822
Usage_flat/PartitionedCallü
Fleet_flat/PartitionedCallPartitionedCall)FleetEmb/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_Fleet_flat_layer_call_and_return_conditional_losses_174902
Fleet_flat/PartitionedCallð
Pc_flat/PartitionedCallPartitionedCall&PcEmb/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_Pc_flat_layer_call_and_return_conditional_losses_174982
Pc_flat/PartitionedCall¬
concate/PartitionedCallPartitionedCallinputs!Cov_flat/PartitionedCall:output:0!Sex_flat/PartitionedCall:output:0"Fuel_flat/PartitionedCall:output:0#Usage_flat/PartitionedCall:output:0#Fleet_flat/PartitionedCall:output:0 Pc_flat/PartitionedCall:output:0*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_concate_layer_call_and_return_conditional_losses_175122
concate/PartitionedCallª
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall concate/PartitionedCall:output:0batch_normalization_4_17514batch_normalization_4_17516batch_normalization_4_17518batch_normalization_4_17520*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_172152/
-batch_normalization_4/StatefulPartitionedCall¼
hidden1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0hidden1_17535hidden1_17537*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_hidden1_layer_call_and_return_conditional_losses_175342!
hidden1/StatefulPartitionedCall®
hidden2/StatefulPartitionedCallStatefulPartitionedCall(hidden1/StatefulPartitionedCall:output:0hidden2_17552hidden2_17554*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_hidden2_layer_call_and_return_conditional_losses_175512!
hidden2/StatefulPartitionedCall®
hidden3/StatefulPartitionedCallStatefulPartitionedCall(hidden2/StatefulPartitionedCall:output:0hidden3_17569hidden3_17571*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_hidden3_layer_call_and_return_conditional_losses_175682!
hidden3/StatefulPartitionedCall®
Network/StatefulPartitionedCallStatefulPartitionedCall(hidden3/StatefulPartitionedCall:output:0network_17585network_17587*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_Network_layer_call_and_return_conditional_losses_175842!
Network/StatefulPartitionedCallñ
Add/PartitionedCallPartitionedCall(Network/StatefulPartitionedCall:output:0inputs_7*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_Add_layer_call_and_return_conditional_losses_175962
Add/PartitionedCall§
 Response/StatefulPartitionedCallStatefulPartitionedCallAdd/PartitionedCall:output:0response_17610response_17612*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_Response_layer_call_and_return_conditional_losses_176092"
 Response/StatefulPartitionedCall¢
IdentityIdentity)Response/StatefulPartitionedCall:output:0^CovEmb/StatefulPartitionedCall!^FleetEmb/StatefulPartitionedCall ^FuelEmb/StatefulPartitionedCall ^Network/StatefulPartitionedCall^PcEmb/StatefulPartitionedCall!^Response/StatefulPartitionedCall^SexEmb/StatefulPartitionedCall!^UsageEmb/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall ^hidden1/StatefulPartitionedCall ^hidden2/StatefulPartitionedCall ^hidden3/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Õ
_input_shapesÃ
À:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : 2@
CovEmb/StatefulPartitionedCallCovEmb/StatefulPartitionedCall2D
 FleetEmb/StatefulPartitionedCall FleetEmb/StatefulPartitionedCall2B
FuelEmb/StatefulPartitionedCallFuelEmb/StatefulPartitionedCall2B
Network/StatefulPartitionedCallNetwork/StatefulPartitionedCall2>
PcEmb/StatefulPartitionedCallPcEmb/StatefulPartitionedCall2D
 Response/StatefulPartitionedCall Response/StatefulPartitionedCall2@
SexEmb/StatefulPartitionedCallSexEmb/StatefulPartitionedCall2D
 UsageEmb/StatefulPartitionedCall UsageEmb/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2B
hidden1/StatefulPartitionedCallhidden1/StatefulPartitionedCall2B
hidden2/StatefulPartitionedCallhidden2/StatefulPartitionedCall2B
hidden3/StatefulPartitionedCallhidden3/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ý
a
E__inference_Usage_flat_layer_call_and_return_conditional_losses_18706

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


'__inference_Network_layer_call_fn_18915

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallò
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_Network_layer_call_and_return_conditional_losses_175842
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


(__inference_Response_layer_call_fn_18947

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_Response_layer_call_and_return_conditional_losses_176092
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
©	

@__inference_PcEmb_layer_call_and_return_conditional_losses_17383

inputs(
embedding_lookup_17377:P
identity¢embedding_lookupù
embedding_lookupResourceGatherembedding_lookup_17377inputs",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*)
_class
loc:@embedding_lookup/17377*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02
embedding_lookupì
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@embedding_lookup/17377*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_lookup/Identity 
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Î	
ó
B__inference_Network_layer_call_and_return_conditional_losses_17584

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Â
D
(__inference_Sex_flat_layer_call_fn_18689

inputs
identityÁ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_Sex_flat_layer_call_and_return_conditional_losses_174662
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
 

ó
B__inference_hidden3_layer_call_and_return_conditional_losses_18887

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Tanh
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
©	

@__inference_PcEmb_layer_call_and_return_conditional_losses_18660

inputs(
embedding_lookup_18654:P
identity¢embedding_lookupù
embedding_lookupResourceGatherembedding_lookup_18654inputs",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*)
_class
loc:@embedding_lookup/18654*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02
embedding_lookupì
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@embedding_lookup/18654*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_lookup/Identity 
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ê
Ö
#__inference_signature_wrapper_18241
coverage

design	
fleet
fuel

loggam

postalcode
sex	
usage
unknown:P
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:


unknown_12:


unknown_13:


unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCalldesigncoveragesexfuelusagefleet
postalcodeloggamunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18*'
Tin 
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__wrapped_model_171912
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Õ
_input_shapesÃ
À:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
Coverage:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameDesign:NJ
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameFleet:MI
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameFuel:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameLogGAM:SO
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
PostalCode:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameSex:NJ
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameUsage
¼

'__inference_concate_layer_call_fn_18756
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
identity
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_concate_layer_call_and_return_conditional_losses_175122
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/2:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/3:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/4:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/5:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/6
Ü
`
D__inference_Fuel_flat_layer_call_and_return_conditional_losses_18695

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

¯
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_17215

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1Û
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
 

ó
B__inference_hidden3_layer_call_and_return_conditional_losses_17568

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Tanh
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
¡
ë
'__inference_model_3_layer_call_fn_18571
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
unknown:P
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:


unknown_12:


unknown_13:


unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:
identity¢StatefulPartitionedCall²
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18*'
Tin 
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_model_3_layer_call_and_return_conditional_losses_179442
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Õ
_input_shapesÃ
À:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/2:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/3:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/4:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/5:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/6:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/7
 

ó
B__inference_hidden2_layer_call_and_return_conditional_losses_17551

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:

identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
Tanh
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Û
_
C__inference_Sex_flat_layer_call_and_return_conditional_losses_18684

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ø
|
(__inference_UsageEmb_layer_call_fn_18635

inputs
unknown:
identity¢StatefulPartitionedCallê
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_UsageEmb_layer_call_and_return_conditional_losses_174092
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
É

²
B__inference_concate_layer_call_and_return_conditional_losses_17512

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis±
concatConcatV2inputsinputs_1inputs_2inputs_3inputs_4inputs_5inputs_6concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
µ
O
#__inference_Add_layer_call_fn_18927
inputs_0
inputs_1
identityÉ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_Add_layer_call_and_return_conditional_losses_175962
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
ô)
é
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_18810

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:2
moments/StopGradient¤
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices²
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay¤
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/sub
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/mul¿
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decayª
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp 
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/sub
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/mulÉ
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1
IdentityIdentitybatchnorm/add_1:z:0^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ä
E
)__inference_Fuel_flat_layer_call_fn_18700

inputs
identityÂ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_Fuel_flat_layer_call_and_return_conditional_losses_174742
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¬­
ã
B__inference_model_3_layer_call_and_return_conditional_losses_18467
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7.
pcemb_embedding_lookup_18357:P1
fleetemb_embedding_lookup_18362:1
usageemb_embedding_lookup_18367:0
fuelemb_embedding_lookup_18372:/
sexemb_embedding_lookup_18377:/
covemb_embedding_lookup_18382:K
=batch_normalization_4_assignmovingavg_readvariableop_resource:M
?batch_normalization_4_assignmovingavg_1_readvariableop_resource:I
;batch_normalization_4_batchnorm_mul_readvariableop_resource:E
7batch_normalization_4_batchnorm_readvariableop_resource:8
&hidden1_matmul_readvariableop_resource:5
'hidden1_biasadd_readvariableop_resource:8
&hidden2_matmul_readvariableop_resource:
5
'hidden2_biasadd_readvariableop_resource:
8
&hidden3_matmul_readvariableop_resource:
5
'hidden3_biasadd_readvariableop_resource:8
&network_matmul_readvariableop_resource:5
'network_biasadd_readvariableop_resource:9
'response_matmul_readvariableop_resource:6
(response_biasadd_readvariableop_resource:
identity¢CovEmb/embedding_lookup¢FleetEmb/embedding_lookup¢FuelEmb/embedding_lookup¢Network/BiasAdd/ReadVariableOp¢Network/MatMul/ReadVariableOp¢PcEmb/embedding_lookup¢Response/BiasAdd/ReadVariableOp¢Response/MatMul/ReadVariableOp¢SexEmb/embedding_lookup¢UsageEmb/embedding_lookup¢%batch_normalization_4/AssignMovingAvg¢4batch_normalization_4/AssignMovingAvg/ReadVariableOp¢'batch_normalization_4/AssignMovingAvg_1¢6batch_normalization_4/AssignMovingAvg_1/ReadVariableOp¢.batch_normalization_4/batchnorm/ReadVariableOp¢2batch_normalization_4/batchnorm/mul/ReadVariableOp¢hidden1/BiasAdd/ReadVariableOp¢hidden1/MatMul/ReadVariableOp¢hidden2/BiasAdd/ReadVariableOp¢hidden2/MatMul/ReadVariableOp¢hidden3/BiasAdd/ReadVariableOp¢hidden3/MatMul/ReadVariableOp
PcEmb/embedding_lookupResourceGatherpcemb_embedding_lookup_18357inputs_6",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*/
_class%
#!loc:@PcEmb/embedding_lookup/18357*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02
PcEmb/embedding_lookup
PcEmb/embedding_lookup/IdentityIdentityPcEmb/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*/
_class%
#!loc:@PcEmb/embedding_lookup/18357*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
PcEmb/embedding_lookup/Identity²
!PcEmb/embedding_lookup/Identity_1Identity(PcEmb/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!PcEmb/embedding_lookup/Identity_1
FleetEmb/embedding_lookupResourceGatherfleetemb_embedding_lookup_18362inputs_5",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*2
_class(
&$loc:@FleetEmb/embedding_lookup/18362*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02
FleetEmb/embedding_lookup
"FleetEmb/embedding_lookup/IdentityIdentity"FleetEmb/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*2
_class(
&$loc:@FleetEmb/embedding_lookup/18362*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"FleetEmb/embedding_lookup/Identity»
$FleetEmb/embedding_lookup/Identity_1Identity+FleetEmb/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$FleetEmb/embedding_lookup/Identity_1
UsageEmb/embedding_lookupResourceGatherusageemb_embedding_lookup_18367inputs_4",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*2
_class(
&$loc:@UsageEmb/embedding_lookup/18367*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02
UsageEmb/embedding_lookup
"UsageEmb/embedding_lookup/IdentityIdentity"UsageEmb/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*2
_class(
&$loc:@UsageEmb/embedding_lookup/18367*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"UsageEmb/embedding_lookup/Identity»
$UsageEmb/embedding_lookup/Identity_1Identity+UsageEmb/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$UsageEmb/embedding_lookup/Identity_1
FuelEmb/embedding_lookupResourceGatherfuelemb_embedding_lookup_18372inputs_3",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*1
_class'
%#loc:@FuelEmb/embedding_lookup/18372*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02
FuelEmb/embedding_lookup
!FuelEmb/embedding_lookup/IdentityIdentity!FuelEmb/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*1
_class'
%#loc:@FuelEmb/embedding_lookup/18372*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!FuelEmb/embedding_lookup/Identity¸
#FuelEmb/embedding_lookup/Identity_1Identity*FuelEmb/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#FuelEmb/embedding_lookup/Identity_1
SexEmb/embedding_lookupResourceGathersexemb_embedding_lookup_18377inputs_2",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*0
_class&
$"loc:@SexEmb/embedding_lookup/18377*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02
SexEmb/embedding_lookup
 SexEmb/embedding_lookup/IdentityIdentity SexEmb/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*0
_class&
$"loc:@SexEmb/embedding_lookup/18377*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 SexEmb/embedding_lookup/Identityµ
"SexEmb/embedding_lookup/Identity_1Identity)SexEmb/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"SexEmb/embedding_lookup/Identity_1
CovEmb/embedding_lookupResourceGathercovemb_embedding_lookup_18382inputs_1",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*0
_class&
$"loc:@CovEmb/embedding_lookup/18382*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02
CovEmb/embedding_lookup
 CovEmb/embedding_lookup/IdentityIdentity CovEmb/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*0
_class&
$"loc:@CovEmb/embedding_lookup/18382*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 CovEmb/embedding_lookup/Identityµ
"CovEmb/embedding_lookup/Identity_1Identity)CovEmb/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"CovEmb/embedding_lookup/Identity_1q
Cov_flat/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
Cov_flat/Const§
Cov_flat/ReshapeReshape+CovEmb/embedding_lookup/Identity_1:output:0Cov_flat/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Cov_flat/Reshapeq
Sex_flat/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
Sex_flat/Const§
Sex_flat/ReshapeReshape+SexEmb/embedding_lookup/Identity_1:output:0Sex_flat/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Sex_flat/Reshapes
Fuel_flat/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
Fuel_flat/Const«
Fuel_flat/ReshapeReshape,FuelEmb/embedding_lookup/Identity_1:output:0Fuel_flat/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Fuel_flat/Reshapeu
Usage_flat/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
Usage_flat/Const¯
Usage_flat/ReshapeReshape-UsageEmb/embedding_lookup/Identity_1:output:0Usage_flat/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Usage_flat/Reshapeu
Fleet_flat/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
Fleet_flat/Const¯
Fleet_flat/ReshapeReshape-FleetEmb/embedding_lookup/Identity_1:output:0Fleet_flat/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Fleet_flat/Reshapeo
Pc_flat/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
Pc_flat/Const£
Pc_flat/ReshapeReshape*PcEmb/embedding_lookup/Identity_1:output:0Pc_flat/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Pc_flat/Reshapel
concate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concate/concat/axisµ
concate/concatConcatV2inputs_0Cov_flat/Reshape:output:0Sex_flat/Reshape:output:0Fuel_flat/Reshape:output:0Usage_flat/Reshape:output:0Fleet_flat/Reshape:output:0Pc_flat/Reshape:output:0concate/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
concate/concat¶
4batch_normalization_4/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_4/moments/mean/reduction_indicesâ
"batch_normalization_4/moments/meanMeanconcate/concat:output:0=batch_normalization_4/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2$
"batch_normalization_4/moments/mean¾
*batch_normalization_4/moments/StopGradientStopGradient+batch_normalization_4/moments/mean:output:0*
T0*
_output_shapes

:2,
*batch_normalization_4/moments/StopGradient÷
/batch_normalization_4/moments/SquaredDifferenceSquaredDifferenceconcate/concat:output:03batch_normalization_4/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/batch_normalization_4/moments/SquaredDifference¾
8batch_normalization_4/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_4/moments/variance/reduction_indices
&batch_normalization_4/moments/varianceMean3batch_normalization_4/moments/SquaredDifference:z:0Abatch_normalization_4/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2(
&batch_normalization_4/moments/varianceÂ
%batch_normalization_4/moments/SqueezeSqueeze+batch_normalization_4/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2'
%batch_normalization_4/moments/SqueezeÊ
'batch_normalization_4/moments/Squeeze_1Squeeze/batch_normalization_4/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2)
'batch_normalization_4/moments/Squeeze_1
+batch_normalization_4/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2-
+batch_normalization_4/AssignMovingAvg/decayæ
4batch_normalization_4/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_4_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype026
4batch_normalization_4/AssignMovingAvg/ReadVariableOpð
)batch_normalization_4/AssignMovingAvg/subSub<batch_normalization_4/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_4/moments/Squeeze:output:0*
T0*
_output_shapes
:2+
)batch_normalization_4/AssignMovingAvg/subç
)batch_normalization_4/AssignMovingAvg/mulMul-batch_normalization_4/AssignMovingAvg/sub:z:04batch_normalization_4/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:2+
)batch_normalization_4/AssignMovingAvg/mul­
%batch_normalization_4/AssignMovingAvgAssignSubVariableOp=batch_normalization_4_assignmovingavg_readvariableop_resource-batch_normalization_4/AssignMovingAvg/mul:z:05^batch_normalization_4/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_4/AssignMovingAvg£
-batch_normalization_4/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2/
-batch_normalization_4/AssignMovingAvg_1/decayì
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_4_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOpø
+batch_normalization_4/AssignMovingAvg_1/subSub>batch_normalization_4/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_4/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2-
+batch_normalization_4/AssignMovingAvg_1/subï
+batch_normalization_4/AssignMovingAvg_1/mulMul/batch_normalization_4/AssignMovingAvg_1/sub:z:06batch_normalization_4/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:2-
+batch_normalization_4/AssignMovingAvg_1/mul·
'batch_normalization_4/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_4_assignmovingavg_1_readvariableop_resource/batch_normalization_4/AssignMovingAvg_1/mul:z:07^batch_normalization_4/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02)
'batch_normalization_4/AssignMovingAvg_1
%batch_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2'
%batch_normalization_4/batchnorm/add/yÚ
#batch_normalization_4/batchnorm/addAddV20batch_normalization_4/moments/Squeeze_1:output:0.batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes
:2%
#batch_normalization_4/batchnorm/add¥
%batch_normalization_4/batchnorm/RsqrtRsqrt'batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes
:2'
%batch_normalization_4/batchnorm/Rsqrtà
2batch_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype024
2batch_normalization_4/batchnorm/mul/ReadVariableOpÝ
#batch_normalization_4/batchnorm/mulMul)batch_normalization_4/batchnorm/Rsqrt:y:0:batch_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2%
#batch_normalization_4/batchnorm/mulÉ
%batch_normalization_4/batchnorm/mul_1Mulconcate/concat:output:0'batch_normalization_4/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%batch_normalization_4/batchnorm/mul_1Ó
%batch_normalization_4/batchnorm/mul_2Mul.batch_normalization_4/moments/Squeeze:output:0'batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes
:2'
%batch_normalization_4/batchnorm/mul_2Ô
.batch_normalization_4/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_4_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype020
.batch_normalization_4/batchnorm/ReadVariableOpÙ
#batch_normalization_4/batchnorm/subSub6batch_normalization_4/batchnorm/ReadVariableOp:value:0)batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2%
#batch_normalization_4/batchnorm/subÝ
%batch_normalization_4/batchnorm/add_1AddV2)batch_normalization_4/batchnorm/mul_1:z:0'batch_normalization_4/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%batch_normalization_4/batchnorm/add_1¥
hidden1/MatMul/ReadVariableOpReadVariableOp&hidden1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
hidden1/MatMul/ReadVariableOp®
hidden1/MatMulMatMul)batch_normalization_4/batchnorm/add_1:z:0%hidden1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
hidden1/MatMul¤
hidden1/BiasAdd/ReadVariableOpReadVariableOp'hidden1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
hidden1/BiasAdd/ReadVariableOp¡
hidden1/BiasAddBiasAddhidden1/MatMul:product:0&hidden1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
hidden1/BiasAddp
hidden1/TanhTanhhidden1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
hidden1/Tanh¥
hidden2/MatMul/ReadVariableOpReadVariableOp&hidden2_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02
hidden2/MatMul/ReadVariableOp
hidden2/MatMulMatMulhidden1/Tanh:y:0%hidden2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
hidden2/MatMul¤
hidden2/BiasAdd/ReadVariableOpReadVariableOp'hidden2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02 
hidden2/BiasAdd/ReadVariableOp¡
hidden2/BiasAddBiasAddhidden2/MatMul:product:0&hidden2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
hidden2/BiasAddp
hidden2/TanhTanhhidden2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
hidden2/Tanh¥
hidden3/MatMul/ReadVariableOpReadVariableOp&hidden3_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02
hidden3/MatMul/ReadVariableOp
hidden3/MatMulMatMulhidden2/Tanh:y:0%hidden3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
hidden3/MatMul¤
hidden3/BiasAdd/ReadVariableOpReadVariableOp'hidden3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
hidden3/BiasAdd/ReadVariableOp¡
hidden3/BiasAddBiasAddhidden3/MatMul:product:0&hidden3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
hidden3/BiasAddp
hidden3/TanhTanhhidden3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
hidden3/Tanh¥
Network/MatMul/ReadVariableOpReadVariableOp&network_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
Network/MatMul/ReadVariableOp
Network/MatMulMatMulhidden3/Tanh:y:0%Network/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Network/MatMul¤
Network/BiasAdd/ReadVariableOpReadVariableOp'network_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
Network/BiasAdd/ReadVariableOp¡
Network/BiasAddBiasAddNetwork/MatMul:product:0&Network/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Network/BiasAddq
Add/addAddV2Network/BiasAdd:output:0inputs_7*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Add/add¨
Response/MatMul/ReadVariableOpReadVariableOp'response_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
Response/MatMul/ReadVariableOp
Response/MatMulMatMulAdd/add:z:0&Response/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Response/MatMul§
Response/BiasAdd/ReadVariableOpReadVariableOp(response_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
Response/BiasAdd/ReadVariableOp¥
Response/BiasAddBiasAddResponse/MatMul:product:0'Response/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Response/BiasAddp
Response/ExpExpResponse/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Response/Expó
IdentityIdentityResponse/Exp:y:0^CovEmb/embedding_lookup^FleetEmb/embedding_lookup^FuelEmb/embedding_lookup^Network/BiasAdd/ReadVariableOp^Network/MatMul/ReadVariableOp^PcEmb/embedding_lookup ^Response/BiasAdd/ReadVariableOp^Response/MatMul/ReadVariableOp^SexEmb/embedding_lookup^UsageEmb/embedding_lookup&^batch_normalization_4/AssignMovingAvg5^batch_normalization_4/AssignMovingAvg/ReadVariableOp(^batch_normalization_4/AssignMovingAvg_17^batch_normalization_4/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_4/batchnorm/ReadVariableOp3^batch_normalization_4/batchnorm/mul/ReadVariableOp^hidden1/BiasAdd/ReadVariableOp^hidden1/MatMul/ReadVariableOp^hidden2/BiasAdd/ReadVariableOp^hidden2/MatMul/ReadVariableOp^hidden3/BiasAdd/ReadVariableOp^hidden3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Õ
_input_shapesÃ
À:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : 22
CovEmb/embedding_lookupCovEmb/embedding_lookup26
FleetEmb/embedding_lookupFleetEmb/embedding_lookup24
FuelEmb/embedding_lookupFuelEmb/embedding_lookup2@
Network/BiasAdd/ReadVariableOpNetwork/BiasAdd/ReadVariableOp2>
Network/MatMul/ReadVariableOpNetwork/MatMul/ReadVariableOp20
PcEmb/embedding_lookupPcEmb/embedding_lookup2B
Response/BiasAdd/ReadVariableOpResponse/BiasAdd/ReadVariableOp2@
Response/MatMul/ReadVariableOpResponse/MatMul/ReadVariableOp22
SexEmb/embedding_lookupSexEmb/embedding_lookup26
UsageEmb/embedding_lookupUsageEmb/embedding_lookup2N
%batch_normalization_4/AssignMovingAvg%batch_normalization_4/AssignMovingAvg2l
4batch_normalization_4/AssignMovingAvg/ReadVariableOp4batch_normalization_4/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_4/AssignMovingAvg_1'batch_normalization_4/AssignMovingAvg_12p
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOp6batch_normalization_4/AssignMovingAvg_1/ReadVariableOp2`
.batch_normalization_4/batchnorm/ReadVariableOp.batch_normalization_4/batchnorm/ReadVariableOp2h
2batch_normalization_4/batchnorm/mul/ReadVariableOp2batch_normalization_4/batchnorm/mul/ReadVariableOp2@
hidden1/BiasAdd/ReadVariableOphidden1/BiasAdd/ReadVariableOp2>
hidden1/MatMul/ReadVariableOphidden1/MatMul/ReadVariableOp2@
hidden2/BiasAdd/ReadVariableOphidden2/BiasAdd/ReadVariableOp2>
hidden2/MatMul/ReadVariableOphidden2/MatMul/ReadVariableOp2@
hidden3/BiasAdd/ReadVariableOphidden3/BiasAdd/ReadVariableOp2>
hidden3/MatMul/ReadVariableOphidden3/MatMul/ReadVariableOp:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/2:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/3:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/4:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/5:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/6:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/7


'__inference_hidden2_layer_call_fn_18876

inputs
unknown:

	unknown_0:

identity¢StatefulPartitionedCallò
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_hidden2_layer_call_and_return_conditional_losses_175512
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Î
j
>__inference_Add_layer_call_and_return_conditional_losses_18921
inputs_0
inputs_1
identityY
addAddV2inputs_0inputs_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add[
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
Æ¢

 __inference__wrapped_model_17191

design
coverage
sex
fuel	
usage	
fleet

postalcode

loggam6
$model_3_pcemb_embedding_lookup_17095:P9
'model_3_fleetemb_embedding_lookup_17100:9
'model_3_usageemb_embedding_lookup_17105:8
&model_3_fuelemb_embedding_lookup_17110:7
%model_3_sexemb_embedding_lookup_17115:7
%model_3_covemb_embedding_lookup_17120:M
?model_3_batch_normalization_4_batchnorm_readvariableop_resource:Q
Cmodel_3_batch_normalization_4_batchnorm_mul_readvariableop_resource:O
Amodel_3_batch_normalization_4_batchnorm_readvariableop_1_resource:O
Amodel_3_batch_normalization_4_batchnorm_readvariableop_2_resource:@
.model_3_hidden1_matmul_readvariableop_resource:=
/model_3_hidden1_biasadd_readvariableop_resource:@
.model_3_hidden2_matmul_readvariableop_resource:
=
/model_3_hidden2_biasadd_readvariableop_resource:
@
.model_3_hidden3_matmul_readvariableop_resource:
=
/model_3_hidden3_biasadd_readvariableop_resource:@
.model_3_network_matmul_readvariableop_resource:=
/model_3_network_biasadd_readvariableop_resource:A
/model_3_response_matmul_readvariableop_resource:>
0model_3_response_biasadd_readvariableop_resource:
identity¢model_3/CovEmb/embedding_lookup¢!model_3/FleetEmb/embedding_lookup¢ model_3/FuelEmb/embedding_lookup¢&model_3/Network/BiasAdd/ReadVariableOp¢%model_3/Network/MatMul/ReadVariableOp¢model_3/PcEmb/embedding_lookup¢'model_3/Response/BiasAdd/ReadVariableOp¢&model_3/Response/MatMul/ReadVariableOp¢model_3/SexEmb/embedding_lookup¢!model_3/UsageEmb/embedding_lookup¢6model_3/batch_normalization_4/batchnorm/ReadVariableOp¢8model_3/batch_normalization_4/batchnorm/ReadVariableOp_1¢8model_3/batch_normalization_4/batchnorm/ReadVariableOp_2¢:model_3/batch_normalization_4/batchnorm/mul/ReadVariableOp¢&model_3/hidden1/BiasAdd/ReadVariableOp¢%model_3/hidden1/MatMul/ReadVariableOp¢&model_3/hidden2/BiasAdd/ReadVariableOp¢%model_3/hidden2/MatMul/ReadVariableOp¢&model_3/hidden3/BiasAdd/ReadVariableOp¢%model_3/hidden3/MatMul/ReadVariableOpµ
model_3/PcEmb/embedding_lookupResourceGather$model_3_pcemb_embedding_lookup_17095
postalcode",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*7
_class-
+)loc:@model_3/PcEmb/embedding_lookup/17095*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02 
model_3/PcEmb/embedding_lookup¤
'model_3/PcEmb/embedding_lookup/IdentityIdentity'model_3/PcEmb/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*7
_class-
+)loc:@model_3/PcEmb/embedding_lookup/17095*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'model_3/PcEmb/embedding_lookup/IdentityÊ
)model_3/PcEmb/embedding_lookup/Identity_1Identity0model_3/PcEmb/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)model_3/PcEmb/embedding_lookup/Identity_1¼
!model_3/FleetEmb/embedding_lookupResourceGather'model_3_fleetemb_embedding_lookup_17100fleet",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*:
_class0
.,loc:@model_3/FleetEmb/embedding_lookup/17100*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02#
!model_3/FleetEmb/embedding_lookup°
*model_3/FleetEmb/embedding_lookup/IdentityIdentity*model_3/FleetEmb/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*:
_class0
.,loc:@model_3/FleetEmb/embedding_lookup/17100*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*model_3/FleetEmb/embedding_lookup/IdentityÓ
,model_3/FleetEmb/embedding_lookup/Identity_1Identity3model_3/FleetEmb/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,model_3/FleetEmb/embedding_lookup/Identity_1¼
!model_3/UsageEmb/embedding_lookupResourceGather'model_3_usageemb_embedding_lookup_17105usage",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*:
_class0
.,loc:@model_3/UsageEmb/embedding_lookup/17105*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02#
!model_3/UsageEmb/embedding_lookup°
*model_3/UsageEmb/embedding_lookup/IdentityIdentity*model_3/UsageEmb/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*:
_class0
.,loc:@model_3/UsageEmb/embedding_lookup/17105*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*model_3/UsageEmb/embedding_lookup/IdentityÓ
,model_3/UsageEmb/embedding_lookup/Identity_1Identity3model_3/UsageEmb/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,model_3/UsageEmb/embedding_lookup/Identity_1·
 model_3/FuelEmb/embedding_lookupResourceGather&model_3_fuelemb_embedding_lookup_17110fuel",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*9
_class/
-+loc:@model_3/FuelEmb/embedding_lookup/17110*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02"
 model_3/FuelEmb/embedding_lookup¬
)model_3/FuelEmb/embedding_lookup/IdentityIdentity)model_3/FuelEmb/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*9
_class/
-+loc:@model_3/FuelEmb/embedding_lookup/17110*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)model_3/FuelEmb/embedding_lookup/IdentityÐ
+model_3/FuelEmb/embedding_lookup/Identity_1Identity2model_3/FuelEmb/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+model_3/FuelEmb/embedding_lookup/Identity_1²
model_3/SexEmb/embedding_lookupResourceGather%model_3_sexemb_embedding_lookup_17115sex",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*8
_class.
,*loc:@model_3/SexEmb/embedding_lookup/17115*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02!
model_3/SexEmb/embedding_lookup¨
(model_3/SexEmb/embedding_lookup/IdentityIdentity(model_3/SexEmb/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*8
_class.
,*loc:@model_3/SexEmb/embedding_lookup/17115*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(model_3/SexEmb/embedding_lookup/IdentityÍ
*model_3/SexEmb/embedding_lookup/Identity_1Identity1model_3/SexEmb/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*model_3/SexEmb/embedding_lookup/Identity_1·
model_3/CovEmb/embedding_lookupResourceGather%model_3_covemb_embedding_lookup_17120coverage",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*8
_class.
,*loc:@model_3/CovEmb/embedding_lookup/17120*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02!
model_3/CovEmb/embedding_lookup¨
(model_3/CovEmb/embedding_lookup/IdentityIdentity(model_3/CovEmb/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*8
_class.
,*loc:@model_3/CovEmb/embedding_lookup/17120*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(model_3/CovEmb/embedding_lookup/IdentityÍ
*model_3/CovEmb/embedding_lookup/Identity_1Identity1model_3/CovEmb/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*model_3/CovEmb/embedding_lookup/Identity_1
model_3/Cov_flat/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
model_3/Cov_flat/ConstÇ
model_3/Cov_flat/ReshapeReshape3model_3/CovEmb/embedding_lookup/Identity_1:output:0model_3/Cov_flat/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_3/Cov_flat/Reshape
model_3/Sex_flat/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
model_3/Sex_flat/ConstÇ
model_3/Sex_flat/ReshapeReshape3model_3/SexEmb/embedding_lookup/Identity_1:output:0model_3/Sex_flat/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_3/Sex_flat/Reshape
model_3/Fuel_flat/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
model_3/Fuel_flat/ConstË
model_3/Fuel_flat/ReshapeReshape4model_3/FuelEmb/embedding_lookup/Identity_1:output:0 model_3/Fuel_flat/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_3/Fuel_flat/Reshape
model_3/Usage_flat/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
model_3/Usage_flat/ConstÏ
model_3/Usage_flat/ReshapeReshape5model_3/UsageEmb/embedding_lookup/Identity_1:output:0!model_3/Usage_flat/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_3/Usage_flat/Reshape
model_3/Fleet_flat/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
model_3/Fleet_flat/ConstÏ
model_3/Fleet_flat/ReshapeReshape5model_3/FleetEmb/embedding_lookup/Identity_1:output:0!model_3/Fleet_flat/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_3/Fleet_flat/Reshape
model_3/Pc_flat/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
model_3/Pc_flat/ConstÃ
model_3/Pc_flat/ReshapeReshape2model_3/PcEmb/embedding_lookup/Identity_1:output:0model_3/Pc_flat/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_3/Pc_flat/Reshape|
model_3/concate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
model_3/concate/concat/axisû
model_3/concate/concatConcatV2design!model_3/Cov_flat/Reshape:output:0!model_3/Sex_flat/Reshape:output:0"model_3/Fuel_flat/Reshape:output:0#model_3/Usage_flat/Reshape:output:0#model_3/Fleet_flat/Reshape:output:0 model_3/Pc_flat/Reshape:output:0$model_3/concate/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_3/concate/concatì
6model_3/batch_normalization_4/batchnorm/ReadVariableOpReadVariableOp?model_3_batch_normalization_4_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype028
6model_3/batch_normalization_4/batchnorm/ReadVariableOp£
-model_3/batch_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2/
-model_3/batch_normalization_4/batchnorm/add/y
+model_3/batch_normalization_4/batchnorm/addAddV2>model_3/batch_normalization_4/batchnorm/ReadVariableOp:value:06model_3/batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes
:2-
+model_3/batch_normalization_4/batchnorm/add½
-model_3/batch_normalization_4/batchnorm/RsqrtRsqrt/model_3/batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes
:2/
-model_3/batch_normalization_4/batchnorm/Rsqrtø
:model_3/batch_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOpCmodel_3_batch_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02<
:model_3/batch_normalization_4/batchnorm/mul/ReadVariableOpý
+model_3/batch_normalization_4/batchnorm/mulMul1model_3/batch_normalization_4/batchnorm/Rsqrt:y:0Bmodel_3/batch_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2-
+model_3/batch_normalization_4/batchnorm/mulé
-model_3/batch_normalization_4/batchnorm/mul_1Mulmodel_3/concate/concat:output:0/model_3/batch_normalization_4/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-model_3/batch_normalization_4/batchnorm/mul_1ò
8model_3/batch_normalization_4/batchnorm/ReadVariableOp_1ReadVariableOpAmodel_3_batch_normalization_4_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8model_3/batch_normalization_4/batchnorm/ReadVariableOp_1ý
-model_3/batch_normalization_4/batchnorm/mul_2Mul@model_3/batch_normalization_4/batchnorm/ReadVariableOp_1:value:0/model_3/batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes
:2/
-model_3/batch_normalization_4/batchnorm/mul_2ò
8model_3/batch_normalization_4/batchnorm/ReadVariableOp_2ReadVariableOpAmodel_3_batch_normalization_4_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02:
8model_3/batch_normalization_4/batchnorm/ReadVariableOp_2û
+model_3/batch_normalization_4/batchnorm/subSub@model_3/batch_normalization_4/batchnorm/ReadVariableOp_2:value:01model_3/batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2-
+model_3/batch_normalization_4/batchnorm/subý
-model_3/batch_normalization_4/batchnorm/add_1AddV21model_3/batch_normalization_4/batchnorm/mul_1:z:0/model_3/batch_normalization_4/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-model_3/batch_normalization_4/batchnorm/add_1½
%model_3/hidden1/MatMul/ReadVariableOpReadVariableOp.model_3_hidden1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02'
%model_3/hidden1/MatMul/ReadVariableOpÎ
model_3/hidden1/MatMulMatMul1model_3/batch_normalization_4/batchnorm/add_1:z:0-model_3/hidden1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_3/hidden1/MatMul¼
&model_3/hidden1/BiasAdd/ReadVariableOpReadVariableOp/model_3_hidden1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&model_3/hidden1/BiasAdd/ReadVariableOpÁ
model_3/hidden1/BiasAddBiasAdd model_3/hidden1/MatMul:product:0.model_3/hidden1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_3/hidden1/BiasAdd
model_3/hidden1/TanhTanh model_3/hidden1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_3/hidden1/Tanh½
%model_3/hidden2/MatMul/ReadVariableOpReadVariableOp.model_3_hidden2_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02'
%model_3/hidden2/MatMul/ReadVariableOpµ
model_3/hidden2/MatMulMatMulmodel_3/hidden1/Tanh:y:0-model_3/hidden2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
model_3/hidden2/MatMul¼
&model_3/hidden2/BiasAdd/ReadVariableOpReadVariableOp/model_3_hidden2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02(
&model_3/hidden2/BiasAdd/ReadVariableOpÁ
model_3/hidden2/BiasAddBiasAdd model_3/hidden2/MatMul:product:0.model_3/hidden2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
model_3/hidden2/BiasAdd
model_3/hidden2/TanhTanh model_3/hidden2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
model_3/hidden2/Tanh½
%model_3/hidden3/MatMul/ReadVariableOpReadVariableOp.model_3_hidden3_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02'
%model_3/hidden3/MatMul/ReadVariableOpµ
model_3/hidden3/MatMulMatMulmodel_3/hidden2/Tanh:y:0-model_3/hidden3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_3/hidden3/MatMul¼
&model_3/hidden3/BiasAdd/ReadVariableOpReadVariableOp/model_3_hidden3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&model_3/hidden3/BiasAdd/ReadVariableOpÁ
model_3/hidden3/BiasAddBiasAdd model_3/hidden3/MatMul:product:0.model_3/hidden3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_3/hidden3/BiasAdd
model_3/hidden3/TanhTanh model_3/hidden3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_3/hidden3/Tanh½
%model_3/Network/MatMul/ReadVariableOpReadVariableOp.model_3_network_matmul_readvariableop_resource*
_output_shapes

:*
dtype02'
%model_3/Network/MatMul/ReadVariableOpµ
model_3/Network/MatMulMatMulmodel_3/hidden3/Tanh:y:0-model_3/Network/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_3/Network/MatMul¼
&model_3/Network/BiasAdd/ReadVariableOpReadVariableOp/model_3_network_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&model_3/Network/BiasAdd/ReadVariableOpÁ
model_3/Network/BiasAddBiasAdd model_3/Network/MatMul:product:0.model_3/Network/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_3/Network/BiasAdd
model_3/Add/addAddV2 model_3/Network/BiasAdd:output:0loggam*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_3/Add/addÀ
&model_3/Response/MatMul/ReadVariableOpReadVariableOp/model_3_response_matmul_readvariableop_resource*
_output_shapes

:*
dtype02(
&model_3/Response/MatMul/ReadVariableOp³
model_3/Response/MatMulMatMulmodel_3/Add/add:z:0.model_3/Response/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_3/Response/MatMul¿
'model_3/Response/BiasAdd/ReadVariableOpReadVariableOp0model_3_response_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'model_3/Response/BiasAdd/ReadVariableOpÅ
model_3/Response/BiasAddBiasAdd!model_3/Response/MatMul:product:0/model_3/Response/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_3/Response/BiasAdd
model_3/Response/ExpExp!model_3/Response/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_3/Response/Exp¿
IdentityIdentitymodel_3/Response/Exp:y:0 ^model_3/CovEmb/embedding_lookup"^model_3/FleetEmb/embedding_lookup!^model_3/FuelEmb/embedding_lookup'^model_3/Network/BiasAdd/ReadVariableOp&^model_3/Network/MatMul/ReadVariableOp^model_3/PcEmb/embedding_lookup(^model_3/Response/BiasAdd/ReadVariableOp'^model_3/Response/MatMul/ReadVariableOp ^model_3/SexEmb/embedding_lookup"^model_3/UsageEmb/embedding_lookup7^model_3/batch_normalization_4/batchnorm/ReadVariableOp9^model_3/batch_normalization_4/batchnorm/ReadVariableOp_19^model_3/batch_normalization_4/batchnorm/ReadVariableOp_2;^model_3/batch_normalization_4/batchnorm/mul/ReadVariableOp'^model_3/hidden1/BiasAdd/ReadVariableOp&^model_3/hidden1/MatMul/ReadVariableOp'^model_3/hidden2/BiasAdd/ReadVariableOp&^model_3/hidden2/MatMul/ReadVariableOp'^model_3/hidden3/BiasAdd/ReadVariableOp&^model_3/hidden3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Õ
_input_shapesÃ
À:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : 2B
model_3/CovEmb/embedding_lookupmodel_3/CovEmb/embedding_lookup2F
!model_3/FleetEmb/embedding_lookup!model_3/FleetEmb/embedding_lookup2D
 model_3/FuelEmb/embedding_lookup model_3/FuelEmb/embedding_lookup2P
&model_3/Network/BiasAdd/ReadVariableOp&model_3/Network/BiasAdd/ReadVariableOp2N
%model_3/Network/MatMul/ReadVariableOp%model_3/Network/MatMul/ReadVariableOp2@
model_3/PcEmb/embedding_lookupmodel_3/PcEmb/embedding_lookup2R
'model_3/Response/BiasAdd/ReadVariableOp'model_3/Response/BiasAdd/ReadVariableOp2P
&model_3/Response/MatMul/ReadVariableOp&model_3/Response/MatMul/ReadVariableOp2B
model_3/SexEmb/embedding_lookupmodel_3/SexEmb/embedding_lookup2F
!model_3/UsageEmb/embedding_lookup!model_3/UsageEmb/embedding_lookup2p
6model_3/batch_normalization_4/batchnorm/ReadVariableOp6model_3/batch_normalization_4/batchnorm/ReadVariableOp2t
8model_3/batch_normalization_4/batchnorm/ReadVariableOp_18model_3/batch_normalization_4/batchnorm/ReadVariableOp_12t
8model_3/batch_normalization_4/batchnorm/ReadVariableOp_28model_3/batch_normalization_4/batchnorm/ReadVariableOp_22x
:model_3/batch_normalization_4/batchnorm/mul/ReadVariableOp:model_3/batch_normalization_4/batchnorm/mul/ReadVariableOp2P
&model_3/hidden1/BiasAdd/ReadVariableOp&model_3/hidden1/BiasAdd/ReadVariableOp2N
%model_3/hidden1/MatMul/ReadVariableOp%model_3/hidden1/MatMul/ReadVariableOp2P
&model_3/hidden2/BiasAdd/ReadVariableOp&model_3/hidden2/BiasAdd/ReadVariableOp2N
%model_3/hidden2/MatMul/ReadVariableOp%model_3/hidden2/MatMul/ReadVariableOp2P
&model_3/hidden3/BiasAdd/ReadVariableOp&model_3/hidden3/BiasAdd/ReadVariableOp2N
%model_3/hidden3/MatMul/ReadVariableOp%model_3/hidden3/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameDesign:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
Coverage:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameSex:MI
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameFuel:NJ
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameUsage:NJ
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameFleet:SO
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
PostalCode:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameLogGAM
ýy
ü
__inference__traced_save_19157
file_prefix0
,savev2_covemb_embeddings_read_readvariableop0
,savev2_sexemb_embeddings_read_readvariableop1
-savev2_fuelemb_embeddings_read_readvariableop2
.savev2_usageemb_embeddings_read_readvariableop2
.savev2_fleetemb_embeddings_read_readvariableop/
+savev2_pcemb_embeddings_read_readvariableop:
6savev2_batch_normalization_4_gamma_read_readvariableop9
5savev2_batch_normalization_4_beta_read_readvariableop@
<savev2_batch_normalization_4_moving_mean_read_readvariableopD
@savev2_batch_normalization_4_moving_variance_read_readvariableop-
)savev2_hidden1_kernel_read_readvariableop+
'savev2_hidden1_bias_read_readvariableop-
)savev2_hidden2_kernel_read_readvariableop+
'savev2_hidden2_bias_read_readvariableop-
)savev2_hidden3_kernel_read_readvariableop+
'savev2_hidden3_bias_read_readvariableop-
)savev2_network_kernel_read_readvariableop+
'savev2_network_bias_read_readvariableop.
*savev2_response_kernel_read_readvariableop,
(savev2_response_bias_read_readvariableop)
%savev2_nadam_iter_read_readvariableop	+
'savev2_nadam_beta_1_read_readvariableop+
'savev2_nadam_beta_2_read_readvariableop*
&savev2_nadam_decay_read_readvariableop2
.savev2_nadam_learning_rate_read_readvariableop3
/savev2_nadam_momentum_cache_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop8
4savev2_nadam_covemb_embeddings_m_read_readvariableop8
4savev2_nadam_sexemb_embeddings_m_read_readvariableop9
5savev2_nadam_fuelemb_embeddings_m_read_readvariableop:
6savev2_nadam_usageemb_embeddings_m_read_readvariableop:
6savev2_nadam_fleetemb_embeddings_m_read_readvariableop7
3savev2_nadam_pcemb_embeddings_m_read_readvariableopB
>savev2_nadam_batch_normalization_4_gamma_m_read_readvariableopA
=savev2_nadam_batch_normalization_4_beta_m_read_readvariableop5
1savev2_nadam_hidden1_kernel_m_read_readvariableop3
/savev2_nadam_hidden1_bias_m_read_readvariableop5
1savev2_nadam_hidden2_kernel_m_read_readvariableop3
/savev2_nadam_hidden2_bias_m_read_readvariableop5
1savev2_nadam_hidden3_kernel_m_read_readvariableop3
/savev2_nadam_hidden3_bias_m_read_readvariableop5
1savev2_nadam_network_kernel_m_read_readvariableop3
/savev2_nadam_network_bias_m_read_readvariableop8
4savev2_nadam_covemb_embeddings_v_read_readvariableop8
4savev2_nadam_sexemb_embeddings_v_read_readvariableop9
5savev2_nadam_fuelemb_embeddings_v_read_readvariableop:
6savev2_nadam_usageemb_embeddings_v_read_readvariableop:
6savev2_nadam_fleetemb_embeddings_v_read_readvariableop7
3savev2_nadam_pcemb_embeddings_v_read_readvariableopB
>savev2_nadam_batch_normalization_4_gamma_v_read_readvariableopA
=savev2_nadam_batch_normalization_4_beta_v_read_readvariableop5
1savev2_nadam_hidden1_kernel_v_read_readvariableop3
/savev2_nadam_hidden1_bias_v_read_readvariableop5
1savev2_nadam_hidden2_kernel_v_read_readvariableop3
/savev2_nadam_hidden2_bias_v_read_readvariableop5
1savev2_nadam_hidden3_kernel_v_read_readvariableop3
/savev2_nadam_hidden3_bias_v_read_readvariableop5
1savev2_nadam_network_kernel_v_read_readvariableop3
/savev2_nadam_network_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameÂ"
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:=*
dtype0*Ô!
valueÊ!BÇ!=B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-2/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-3/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-4/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-5/embeddings/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/momentum_cache/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-2/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-3/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-4/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-5/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-2/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-3/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-4/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-5/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:=*
dtype0*
valueB=B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0,savev2_covemb_embeddings_read_readvariableop,savev2_sexemb_embeddings_read_readvariableop-savev2_fuelemb_embeddings_read_readvariableop.savev2_usageemb_embeddings_read_readvariableop.savev2_fleetemb_embeddings_read_readvariableop+savev2_pcemb_embeddings_read_readvariableop6savev2_batch_normalization_4_gamma_read_readvariableop5savev2_batch_normalization_4_beta_read_readvariableop<savev2_batch_normalization_4_moving_mean_read_readvariableop@savev2_batch_normalization_4_moving_variance_read_readvariableop)savev2_hidden1_kernel_read_readvariableop'savev2_hidden1_bias_read_readvariableop)savev2_hidden2_kernel_read_readvariableop'savev2_hidden2_bias_read_readvariableop)savev2_hidden3_kernel_read_readvariableop'savev2_hidden3_bias_read_readvariableop)savev2_network_kernel_read_readvariableop'savev2_network_bias_read_readvariableop*savev2_response_kernel_read_readvariableop(savev2_response_bias_read_readvariableop%savev2_nadam_iter_read_readvariableop'savev2_nadam_beta_1_read_readvariableop'savev2_nadam_beta_2_read_readvariableop&savev2_nadam_decay_read_readvariableop.savev2_nadam_learning_rate_read_readvariableop/savev2_nadam_momentum_cache_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop4savev2_nadam_covemb_embeddings_m_read_readvariableop4savev2_nadam_sexemb_embeddings_m_read_readvariableop5savev2_nadam_fuelemb_embeddings_m_read_readvariableop6savev2_nadam_usageemb_embeddings_m_read_readvariableop6savev2_nadam_fleetemb_embeddings_m_read_readvariableop3savev2_nadam_pcemb_embeddings_m_read_readvariableop>savev2_nadam_batch_normalization_4_gamma_m_read_readvariableop=savev2_nadam_batch_normalization_4_beta_m_read_readvariableop1savev2_nadam_hidden1_kernel_m_read_readvariableop/savev2_nadam_hidden1_bias_m_read_readvariableop1savev2_nadam_hidden2_kernel_m_read_readvariableop/savev2_nadam_hidden2_bias_m_read_readvariableop1savev2_nadam_hidden3_kernel_m_read_readvariableop/savev2_nadam_hidden3_bias_m_read_readvariableop1savev2_nadam_network_kernel_m_read_readvariableop/savev2_nadam_network_bias_m_read_readvariableop4savev2_nadam_covemb_embeddings_v_read_readvariableop4savev2_nadam_sexemb_embeddings_v_read_readvariableop5savev2_nadam_fuelemb_embeddings_v_read_readvariableop6savev2_nadam_usageemb_embeddings_v_read_readvariableop6savev2_nadam_fleetemb_embeddings_v_read_readvariableop3savev2_nadam_pcemb_embeddings_v_read_readvariableop>savev2_nadam_batch_normalization_4_gamma_v_read_readvariableop=savev2_nadam_batch_normalization_4_beta_v_read_readvariableop1savev2_nadam_hidden1_kernel_v_read_readvariableop/savev2_nadam_hidden1_bias_v_read_readvariableop1savev2_nadam_hidden2_kernel_v_read_readvariableop/savev2_nadam_hidden2_bias_v_read_readvariableop1savev2_nadam_hidden3_kernel_v_read_readvariableop/savev2_nadam_hidden3_bias_v_read_readvariableop1savev2_nadam_network_kernel_v_read_readvariableop/savev2_nadam_network_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *K
dtypesA
?2=	2
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*Ý
_input_shapesË
È: ::::::P:::::::
:
:
:::::: : : : : : : : ::::::P:::::
:
:
:::::::::P:::::
:
:
:::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

:P: 

_output_shapes
:: 

_output_shapes
:: 	

_output_shapes
:: 


_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:
: 

_output_shapes
:
:$ 

_output_shapes

:
: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$  

_output_shapes

::$! 

_output_shapes

::$" 

_output_shapes

:P: #

_output_shapes
:: $

_output_shapes
::$% 

_output_shapes

:: &

_output_shapes
::$' 

_output_shapes

:
: (

_output_shapes
:
:$) 

_output_shapes

:
: *

_output_shapes
::$+ 

_output_shapes

:: ,

_output_shapes
::$- 

_output_shapes

::$. 

_output_shapes

::$/ 

_output_shapes

::$0 

_output_shapes

::$1 

_output_shapes

::$2 

_output_shapes

:P: 3

_output_shapes
:: 4

_output_shapes
::$5 

_output_shapes

:: 6

_output_shapes
::$7 

_output_shapes

:
: 8

_output_shapes
:
:$9 

_output_shapes

:
: :

_output_shapes
::$; 

_output_shapes

:: <

_output_shapes
::=

_output_shapes
: 
«	

B__inference_FuelEmb_layer_call_and_return_conditional_losses_18612

inputs(
embedding_lookup_18606:
identity¢embedding_lookupù
embedding_lookupResourceGatherembedding_lookup_18606inputs",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*)
_class
loc:@embedding_lookup/18606*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02
embedding_lookupì
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@embedding_lookup/18606*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_lookup/Identity 
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


'__inference_hidden1_layer_call_fn_18856

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallò
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_hidden1_layer_call_and_return_conditional_losses_175342
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs"ÌL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ä
serving_default°
=
Coverage1
serving_default_Coverage:0ÿÿÿÿÿÿÿÿÿ
9
Design/
serving_default_Design:0ÿÿÿÿÿÿÿÿÿ
7
Fleet.
serving_default_Fleet:0ÿÿÿÿÿÿÿÿÿ
5
Fuel-
serving_default_Fuel:0ÿÿÿÿÿÿÿÿÿ
9
LogGAM/
serving_default_LogGAM:0ÿÿÿÿÿÿÿÿÿ
A

PostalCode3
serving_default_PostalCode:0ÿÿÿÿÿÿÿÿÿ
3
Sex,
serving_default_Sex:0ÿÿÿÿÿÿÿÿÿ
7
Usage.
serving_default_Usage:0ÿÿÿÿÿÿÿÿÿ<
Response0
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:º
ÅÆ
layer-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer_with_weights-0
layer-6
layer_with_weights-1
layer-7
	layer_with_weights-2
	layer-8

layer_with_weights-3

layer-9
layer_with_weights-4
layer-10
layer_with_weights-5
layer-11
layer-12
layer-13
layer-14
layer-15
layer-16
layer-17
layer-18
layer-19
layer_with_weights-6
layer-20
layer_with_weights-7
layer-21
layer_with_weights-8
layer-22
layer_with_weights-9
layer-23
layer_with_weights-10
layer-24
layer-25
layer-26
layer_with_weights-11
layer-27
	optimizer
	variables
trainable_variables
 regularization_losses
!	keras_api
"
signatures
+&call_and_return_all_conditional_losses
_default_save_signature
__call__"¾¿
_tf_keras_network¡¿{"name": "model_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "model_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "int32", "sparse": false, "ragged": false, "name": "Coverage"}, "name": "Coverage", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "int32", "sparse": false, "ragged": false, "name": "Sex"}, "name": "Sex", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "int32", "sparse": false, "ragged": false, "name": "Fuel"}, "name": "Fuel", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "int32", "sparse": false, "ragged": false, "name": "Usage"}, "name": "Usage", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "int32", "sparse": false, "ragged": false, "name": "Fleet"}, "name": "Fleet", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "int32", "sparse": false, "ragged": false, "name": "PostalCode"}, "name": "PostalCode", "inbound_nodes": []}, {"class_name": "Embedding", "config": {"name": "CovEmb", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "input_dim": 3, "output_dim": 2, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 1}, "name": "CovEmb", "inbound_nodes": [[["Coverage", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "SexEmb", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "input_dim": 2, "output_dim": 1, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 1}, "name": "SexEmb", "inbound_nodes": [[["Sex", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "FuelEmb", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "input_dim": 2, "output_dim": 1, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 1}, "name": "FuelEmb", "inbound_nodes": [[["Fuel", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "UsageEmb", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "input_dim": 2, "output_dim": 1, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 1}, "name": "UsageEmb", "inbound_nodes": [[["Usage", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "FleetEmb", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "input_dim": 2, "output_dim": 1, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 1}, "name": "FleetEmb", "inbound_nodes": [[["Fleet", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "PcEmb", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "input_dim": 80, "output_dim": 1, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 1}, "name": "PcEmb", "inbound_nodes": [[["PostalCode", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "Design"}, "name": "Design", "inbound_nodes": []}, {"class_name": "Flatten", "config": {"name": "Cov_flat", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "Cov_flat", "inbound_nodes": [[["CovEmb", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "Sex_flat", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "Sex_flat", "inbound_nodes": [[["SexEmb", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "Fuel_flat", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "Fuel_flat", "inbound_nodes": [[["FuelEmb", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "Usage_flat", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "Usage_flat", "inbound_nodes": [[["UsageEmb", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "Fleet_flat", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "Fleet_flat", "inbound_nodes": [[["FleetEmb", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "Pc_flat", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "Pc_flat", "inbound_nodes": [[["PcEmb", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concate", "trainable": null, "dtype": "float32", "axis": -1}, "name": "concate", "inbound_nodes": [[["Design", 0, 0, {}], ["Cov_flat", 0, 0, {}], ["Sex_flat", 0, 0, {}], ["Fuel_flat", 0, 0, {}], ["Usage_flat", 0, 0, {}], ["Fleet_flat", 0, 0, {}], ["Pc_flat", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_4", "inbound_nodes": [[["concate", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "hidden1", "trainable": true, "dtype": "float32", "units": 15, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "hidden1", "inbound_nodes": [[["batch_normalization_4", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "hidden2", "trainable": true, "dtype": "float32", "units": 10, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "hidden2", "inbound_nodes": [[["hidden1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "hidden3", "trainable": true, "dtype": "float32", "units": 5, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "hidden3", "inbound_nodes": [[["hidden2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "Network", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Network", "inbound_nodes": [[["hidden3", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "LogGAM"}, "name": "LogGAM", "inbound_nodes": []}, {"class_name": "Add", "config": {"name": "Add", "trainable": null, "dtype": "float32"}, "name": "Add", "inbound_nodes": [[["Network", 0, 0, {}], ["LogGAM", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "Response", "trainable": false, "dtype": "float32", "units": 1, "activation": "python_function", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Response", "inbound_nodes": [[["Add", 0, 0, {}]]]}], "input_layers": [["Design", 0, 0], ["Coverage", 0, 0], ["Sex", 0, 0], ["Fuel", 0, 0], ["Usage", 0, 0], ["Fleet", 0, 0], ["PostalCode", 0, 0], ["LogGAM", 0, 0]], "output_layers": [["Response", 0, 0]]}, "shared_object_id": 48, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 4]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": [{"class_name": "TensorShape", "items": [null, 4]}, {"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 1]}], "is_graph_network": true, "save_spec": [{"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 4]}, "float32", "Design"]}, {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "int32", "Coverage"]}, {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "int32", "Sex"]}, {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "int32", "Fuel"]}, {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "int32", "Usage"]}, {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "int32", "Fleet"]}, {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "int32", "PostalCode"]}, {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "LogGAM"]}], "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "int32", "sparse": false, "ragged": false, "name": "Coverage"}, "name": "Coverage", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "int32", "sparse": false, "ragged": false, "name": "Sex"}, "name": "Sex", "inbound_nodes": [], "shared_object_id": 1}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "int32", "sparse": false, "ragged": false, "name": "Fuel"}, "name": "Fuel", "inbound_nodes": [], "shared_object_id": 2}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "int32", "sparse": false, "ragged": false, "name": "Usage"}, "name": "Usage", "inbound_nodes": [], "shared_object_id": 3}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "int32", "sparse": false, "ragged": false, "name": "Fleet"}, "name": "Fleet", "inbound_nodes": [], "shared_object_id": 4}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "int32", "sparse": false, "ragged": false, "name": "PostalCode"}, "name": "PostalCode", "inbound_nodes": [], "shared_object_id": 5}, {"class_name": "Embedding", "config": {"name": "CovEmb", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "input_dim": 3, "output_dim": 2, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}, "shared_object_id": 6}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 1}, "name": "CovEmb", "inbound_nodes": [[["Coverage", 0, 0, {}]]], "shared_object_id": 7}, {"class_name": "Embedding", "config": {"name": "SexEmb", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "input_dim": 2, "output_dim": 1, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}, "shared_object_id": 8}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 1}, "name": "SexEmb", "inbound_nodes": [[["Sex", 0, 0, {}]]], "shared_object_id": 9}, {"class_name": "Embedding", "config": {"name": "FuelEmb", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "input_dim": 2, "output_dim": 1, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}, "shared_object_id": 10}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 1}, "name": "FuelEmb", "inbound_nodes": [[["Fuel", 0, 0, {}]]], "shared_object_id": 11}, {"class_name": "Embedding", "config": {"name": "UsageEmb", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "input_dim": 2, "output_dim": 1, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}, "shared_object_id": 12}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 1}, "name": "UsageEmb", "inbound_nodes": [[["Usage", 0, 0, {}]]], "shared_object_id": 13}, {"class_name": "Embedding", "config": {"name": "FleetEmb", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "input_dim": 2, "output_dim": 1, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}, "shared_object_id": 14}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 1}, "name": "FleetEmb", "inbound_nodes": [[["Fleet", 0, 0, {}]]], "shared_object_id": 15}, {"class_name": "Embedding", "config": {"name": "PcEmb", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "input_dim": 80, "output_dim": 1, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}, "shared_object_id": 16}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 1}, "name": "PcEmb", "inbound_nodes": [[["PostalCode", 0, 0, {}]]], "shared_object_id": 17}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "Design"}, "name": "Design", "inbound_nodes": [], "shared_object_id": 18}, {"class_name": "Flatten", "config": {"name": "Cov_flat", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "Cov_flat", "inbound_nodes": [[["CovEmb", 0, 0, {}]]], "shared_object_id": 19}, {"class_name": "Flatten", "config": {"name": "Sex_flat", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "Sex_flat", "inbound_nodes": [[["SexEmb", 0, 0, {}]]], "shared_object_id": 20}, {"class_name": "Flatten", "config": {"name": "Fuel_flat", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "Fuel_flat", "inbound_nodes": [[["FuelEmb", 0, 0, {}]]], "shared_object_id": 21}, {"class_name": "Flatten", "config": {"name": "Usage_flat", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "Usage_flat", "inbound_nodes": [[["UsageEmb", 0, 0, {}]]], "shared_object_id": 22}, {"class_name": "Flatten", "config": {"name": "Fleet_flat", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "Fleet_flat", "inbound_nodes": [[["FleetEmb", 0, 0, {}]]], "shared_object_id": 23}, {"class_name": "Flatten", "config": {"name": "Pc_flat", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "Pc_flat", "inbound_nodes": [[["PcEmb", 0, 0, {}]]], "shared_object_id": 24}, {"class_name": "Concatenate", "config": {"name": "concate", "trainable": null, "dtype": "float32", "axis": -1}, "name": "concate", "inbound_nodes": [[["Design", 0, 0, {}], ["Cov_flat", 0, 0, {}], ["Sex_flat", 0, 0, {}], ["Fuel_flat", 0, 0, {}], ["Usage_flat", 0, 0, {}], ["Fleet_flat", 0, 0, {}], ["Pc_flat", 0, 0, {}]]], "shared_object_id": 25}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 26}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 27}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 28}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 29}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_4", "inbound_nodes": [[["concate", 0, 0, {}]]], "shared_object_id": 30}, {"class_name": "Dense", "config": {"name": "hidden1", "trainable": true, "dtype": "float32", "units": 15, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 31}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 32}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "hidden1", "inbound_nodes": [[["batch_normalization_4", 0, 0, {}]]], "shared_object_id": 33}, {"class_name": "Dense", "config": {"name": "hidden2", "trainable": true, "dtype": "float32", "units": 10, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 34}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 35}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "hidden2", "inbound_nodes": [[["hidden1", 0, 0, {}]]], "shared_object_id": 36}, {"class_name": "Dense", "config": {"name": "hidden3", "trainable": true, "dtype": "float32", "units": 5, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 37}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 38}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "hidden3", "inbound_nodes": [[["hidden2", 0, 0, {}]]], "shared_object_id": 39}, {"class_name": "Dense", "config": {"name": "Network", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 40}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 41}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Network", "inbound_nodes": [[["hidden3", 0, 0, {}]]], "shared_object_id": 42}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "LogGAM"}, "name": "LogGAM", "inbound_nodes": [], "shared_object_id": 43}, {"class_name": "Add", "config": {"name": "Add", "trainable": null, "dtype": "float32"}, "name": "Add", "inbound_nodes": [[["Network", 0, 0, {}], ["LogGAM", 0, 0, {}]]], "shared_object_id": 44}, {"class_name": "Dense", "config": {"name": "Response", "trainable": false, "dtype": "float32", "units": 1, "activation": "python_function", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 45}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 46}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Response", "inbound_nodes": [[["Add", 0, 0, {}]]], "shared_object_id": 47}], "input_layers": [["Design", 0, 0], ["Coverage", 0, 0], ["Sex", 0, 0], ["Fuel", 0, 0], ["Usage", 0, 0], ["Fleet", 0, 0], ["PostalCode", 0, 0], ["LogGAM", 0, 0]], "output_layers": [["Response", 0, 0]]}}, "training_config": {"loss": "poisson", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Nadam", "config": {"name": "Nadam", "learning_rate": 0.0020000000949949026, "decay": 0.004, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07}}}}
ç"ä
_tf_keras_input_layerÄ{"class_name": "InputLayer", "name": "Coverage", "dtype": "int32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "int32", "sparse": false, "ragged": false, "name": "Coverage"}}
Ý"Ú
_tf_keras_input_layerº{"class_name": "InputLayer", "name": "Sex", "dtype": "int32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "int32", "sparse": false, "ragged": false, "name": "Sex"}}
ß"Ü
_tf_keras_input_layer¼{"class_name": "InputLayer", "name": "Fuel", "dtype": "int32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "int32", "sparse": false, "ragged": false, "name": "Fuel"}}
á"Þ
_tf_keras_input_layer¾{"class_name": "InputLayer", "name": "Usage", "dtype": "int32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "int32", "sparse": false, "ragged": false, "name": "Usage"}}
á"Þ
_tf_keras_input_layer¾{"class_name": "InputLayer", "name": "Fleet", "dtype": "int32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "int32", "sparse": false, "ragged": false, "name": "Fleet"}}
ë"è
_tf_keras_input_layerÈ{"class_name": "InputLayer", "name": "PostalCode", "dtype": "int32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "int32", "sparse": false, "ragged": false, "name": "PostalCode"}}
ó
#
embeddings
$trainable_variables
%	variables
&regularization_losses
'	keras_api
+&call_and_return_all_conditional_losses
 __call__"Ò
_tf_keras_layer¸{"name": "CovEmb", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "stateful": false, "must_restore_from_config": false, "class_name": "Embedding", "config": {"name": "CovEmb", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "input_dim": 3, "output_dim": 2, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}, "shared_object_id": 6}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 1}, "inbound_nodes": [[["Coverage", 0, 0, {}]]], "shared_object_id": 7, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1]}}
î
(
embeddings
)trainable_variables
*	variables
+regularization_losses
,	keras_api
+¡&call_and_return_all_conditional_losses
¢__call__"Í
_tf_keras_layer³{"name": "SexEmb", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "stateful": false, "must_restore_from_config": false, "class_name": "Embedding", "config": {"name": "SexEmb", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "input_dim": 2, "output_dim": 1, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}, "shared_object_id": 8}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 1}, "inbound_nodes": [[["Sex", 0, 0, {}]]], "shared_object_id": 9, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1]}}
ó
-
embeddings
.trainable_variables
/	variables
0regularization_losses
1	keras_api
+£&call_and_return_all_conditional_losses
¤__call__"Ò
_tf_keras_layer¸{"name": "FuelEmb", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "stateful": false, "must_restore_from_config": false, "class_name": "Embedding", "config": {"name": "FuelEmb", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "input_dim": 2, "output_dim": 1, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}, "shared_object_id": 10}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 1}, "inbound_nodes": [[["Fuel", 0, 0, {}]]], "shared_object_id": 11, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1]}}
ö
2
embeddings
3trainable_variables
4	variables
5regularization_losses
6	keras_api
+¥&call_and_return_all_conditional_losses
¦__call__"Õ
_tf_keras_layer»{"name": "UsageEmb", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "stateful": false, "must_restore_from_config": false, "class_name": "Embedding", "config": {"name": "UsageEmb", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "input_dim": 2, "output_dim": 1, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}, "shared_object_id": 12}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 1}, "inbound_nodes": [[["Usage", 0, 0, {}]]], "shared_object_id": 13, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1]}}
ö
7
embeddings
8trainable_variables
9	variables
:regularization_losses
;	keras_api
+§&call_and_return_all_conditional_losses
¨__call__"Õ
_tf_keras_layer»{"name": "FleetEmb", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "stateful": false, "must_restore_from_config": false, "class_name": "Embedding", "config": {"name": "FleetEmb", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "input_dim": 2, "output_dim": 1, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}, "shared_object_id": 14}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 1}, "inbound_nodes": [[["Fleet", 0, 0, {}]]], "shared_object_id": 15, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1]}}
ö
<
embeddings
=trainable_variables
>	variables
?regularization_losses
@	keras_api
+©&call_and_return_all_conditional_losses
ª__call__"Õ
_tf_keras_layer»{"name": "PcEmb", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "stateful": false, "must_restore_from_config": false, "class_name": "Embedding", "config": {"name": "PcEmb", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "input_dim": 80, "output_dim": 1, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}, "shared_object_id": 16}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 1}, "inbound_nodes": [[["PostalCode", 0, 0, {}]]], "shared_object_id": 17, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1]}}
ç"ä
_tf_keras_input_layerÄ{"class_name": "InputLayer", "name": "Design", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "Design"}}
Á
Atrainable_variables
B	variables
Cregularization_losses
D	keras_api
+«&call_and_return_all_conditional_losses
¬__call__"°
_tf_keras_layer{"name": "Cov_flat", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "Cov_flat", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "inbound_nodes": [[["CovEmb", 0, 0, {}]]], "shared_object_id": 19, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 57}}
Á
Etrainable_variables
F	variables
Gregularization_losses
H	keras_api
+­&call_and_return_all_conditional_losses
®__call__"°
_tf_keras_layer{"name": "Sex_flat", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "Sex_flat", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "inbound_nodes": [[["SexEmb", 0, 0, {}]]], "shared_object_id": 20, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 58}}
Ä
Itrainable_variables
J	variables
Kregularization_losses
L	keras_api
+¯&call_and_return_all_conditional_losses
°__call__"³
_tf_keras_layer{"name": "Fuel_flat", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "Fuel_flat", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "inbound_nodes": [[["FuelEmb", 0, 0, {}]]], "shared_object_id": 21, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 59}}
Ç
Mtrainable_variables
N	variables
Oregularization_losses
P	keras_api
+±&call_and_return_all_conditional_losses
²__call__"¶
_tf_keras_layer{"name": "Usage_flat", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "Usage_flat", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "inbound_nodes": [[["UsageEmb", 0, 0, {}]]], "shared_object_id": 22, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 60}}
Ç
Qtrainable_variables
R	variables
Sregularization_losses
T	keras_api
+³&call_and_return_all_conditional_losses
´__call__"¶
_tf_keras_layer{"name": "Fleet_flat", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "Fleet_flat", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "inbound_nodes": [[["FleetEmb", 0, 0, {}]]], "shared_object_id": 23, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 61}}
¾
Utrainable_variables
V	variables
Wregularization_losses
X	keras_api
+µ&call_and_return_all_conditional_losses
¶__call__"­
_tf_keras_layer{"name": "Pc_flat", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "Pc_flat", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "inbound_nodes": [[["PcEmb", 0, 0, {}]]], "shared_object_id": 24, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 62}}

Ytrainable_variables
Z	variables
[regularization_losses
\	keras_api
+·&call_and_return_all_conditional_losses
¸__call__"
_tf_keras_layerì{"name": "concate", "trainable": null, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Concatenate", "config": {"name": "concate", "trainable": null, "dtype": "float32", "axis": -1}, "inbound_nodes": [[["Design", 0, 0, {}], ["Cov_flat", 0, 0, {}], ["Sex_flat", 0, 0, {}], ["Fuel_flat", 0, 0, {}], ["Usage_flat", 0, 0, {}], ["Fleet_flat", 0, 0, {}], ["Pc_flat", 0, 0, {}]]], "shared_object_id": 25, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 4]}, {"class_name": "TensorShape", "items": [null, 2]}, {"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 1]}]}
ð

]axis
	^gamma
_beta
`moving_mean
amoving_variance
btrainable_variables
c	variables
dregularization_losses
e	keras_api
+¹&call_and_return_all_conditional_losses
º__call__"	
_tf_keras_layer	{"name": "batch_normalization_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 26}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 27}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 28}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 29}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "inbound_nodes": [[["concate", 0, 0, {}]]], "shared_object_id": 30, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 11}}, "shared_object_id": 63}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 11]}}
	

fkernel
gbias
htrainable_variables
i	variables
jregularization_losses
k	keras_api
+»&call_and_return_all_conditional_losses
¼__call__"å
_tf_keras_layerË{"name": "hidden1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "hidden1", "trainable": true, "dtype": "float32", "units": 15, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 31}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 32}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["batch_normalization_4", 0, 0, {}]]], "shared_object_id": 33, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 11}}, "shared_object_id": 64}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 11]}}
þ

lkernel
mbias
ntrainable_variables
o	variables
pregularization_losses
q	keras_api
+½&call_and_return_all_conditional_losses
¾__call__"×
_tf_keras_layer½{"name": "hidden2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "hidden2", "trainable": true, "dtype": "float32", "units": 10, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 34}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 35}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["hidden1", 0, 0, {}]]], "shared_object_id": 36, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 15}}, "shared_object_id": 65}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 15]}}
ý

rkernel
sbias
ttrainable_variables
u	variables
vregularization_losses
w	keras_api
+¿&call_and_return_all_conditional_losses
À__call__"Ö
_tf_keras_layer¼{"name": "hidden3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "hidden3", "trainable": true, "dtype": "float32", "units": 5, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 37}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 38}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["hidden2", 0, 0, {}]]], "shared_object_id": 39, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 10}}, "shared_object_id": 66}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10]}}
ý

xkernel
ybias
ztrainable_variables
{	variables
|regularization_losses
}	keras_api
+Á&call_and_return_all_conditional_losses
Â__call__"Ö
_tf_keras_layer¼{"name": "Network", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "Network", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 40}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 41}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["hidden3", 0, 0, {}]]], "shared_object_id": 42, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 5}}, "shared_object_id": 67}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 5]}}
ç"ä
_tf_keras_input_layerÄ{"class_name": "InputLayer", "name": "LogGAM", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "LogGAM"}}

~trainable_variables
	variables
regularization_losses
	keras_api
+Ã&call_and_return_all_conditional_losses
Ä__call__"î
_tf_keras_layerÔ{"name": "Add", "trainable": null, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Add", "config": {"name": "Add", "trainable": null, "dtype": "float32"}, "inbound_nodes": [[["Network", 0, 0, {}], ["LogGAM", 0, 0, {}]]], "shared_object_id": 44, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 1]}]}
	
kernel
	bias
trainable_variables
	variables
regularization_losses
	keras_api
+Å&call_and_return_all_conditional_losses
Æ__call__"ß
_tf_keras_layerÅ{"name": "Response", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "Response", "trainable": false, "dtype": "float32", "units": 1, "activation": "python_function", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 45}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 46}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["Add", 0, 0, {}]]], "shared_object_id": 47, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1}}, "shared_object_id": 68}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1]}}
­
	iter
beta_1
beta_2

decay
learning_rate
momentum_cache#mü(mý-mþ2mÿ7m<m^m_mfmgmlmmmrmsmxmym#v(v-v2v7v<v^v_vfvgvlvmvrvsvxvyv"
	optimizer
¸
#0
(1
-2
23
74
<5
^6
_7
`8
a9
f10
g11
l12
m13
r14
s15
x16
y17
18
19"
trackable_list_wrapper

#0
(1
-2
23
74
<5
^6
_7
f8
g9
l10
m11
r12
s13
x14
y15"
trackable_list_wrapper
 "
trackable_list_wrapper
Ó
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
 regularization_losses
metrics
non_trainable_variables
layers
__call__
_default_save_signature
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
-
Çserving_default"
signature_map
#:!2CovEmb/embeddings
'
#0"
trackable_list_wrapper
'
#0"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
 layer_regularization_losses
layer_metrics
$trainable_variables
%	variables
&regularization_losses
metrics
non_trainable_variables
layers
 __call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
#:!2SexEmb/embeddings
'
(0"
trackable_list_wrapper
'
(0"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
 layer_regularization_losses
layer_metrics
)trainable_variables
*	variables
+regularization_losses
metrics
non_trainable_variables
layers
¢__call__
+¡&call_and_return_all_conditional_losses
'¡"call_and_return_conditional_losses"
_generic_user_object
$:"2FuelEmb/embeddings
'
-0"
trackable_list_wrapper
'
-0"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
 layer_regularization_losses
layer_metrics
.trainable_variables
/	variables
0regularization_losses
metrics
 non_trainable_variables
¡layers
¤__call__
+£&call_and_return_all_conditional_losses
'£"call_and_return_conditional_losses"
_generic_user_object
%:#2UsageEmb/embeddings
'
20"
trackable_list_wrapper
'
20"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
 ¢layer_regularization_losses
£layer_metrics
3trainable_variables
4	variables
5regularization_losses
¤metrics
¥non_trainable_variables
¦layers
¦__call__
+¥&call_and_return_all_conditional_losses
'¥"call_and_return_conditional_losses"
_generic_user_object
%:#2FleetEmb/embeddings
'
70"
trackable_list_wrapper
'
70"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
 §layer_regularization_losses
¨layer_metrics
8trainable_variables
9	variables
:regularization_losses
©metrics
ªnon_trainable_variables
«layers
¨__call__
+§&call_and_return_all_conditional_losses
'§"call_and_return_conditional_losses"
_generic_user_object
": P2PcEmb/embeddings
'
<0"
trackable_list_wrapper
'
<0"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
 ¬layer_regularization_losses
­layer_metrics
=trainable_variables
>	variables
?regularization_losses
®metrics
¯non_trainable_variables
°layers
ª__call__
+©&call_and_return_all_conditional_losses
'©"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
 ±layer_regularization_losses
²layer_metrics
Atrainable_variables
B	variables
Cregularization_losses
³metrics
´non_trainable_variables
µlayers
¬__call__
+«&call_and_return_all_conditional_losses
'«"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
 ¶layer_regularization_losses
·layer_metrics
Etrainable_variables
F	variables
Gregularization_losses
¸metrics
¹non_trainable_variables
ºlayers
®__call__
+­&call_and_return_all_conditional_losses
'­"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
 »layer_regularization_losses
¼layer_metrics
Itrainable_variables
J	variables
Kregularization_losses
½metrics
¾non_trainable_variables
¿layers
°__call__
+¯&call_and_return_all_conditional_losses
'¯"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
 Àlayer_regularization_losses
Álayer_metrics
Mtrainable_variables
N	variables
Oregularization_losses
Âmetrics
Ãnon_trainable_variables
Älayers
²__call__
+±&call_and_return_all_conditional_losses
'±"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
 Ålayer_regularization_losses
Ælayer_metrics
Qtrainable_variables
R	variables
Sregularization_losses
Çmetrics
Ènon_trainable_variables
Élayers
´__call__
+³&call_and_return_all_conditional_losses
'³"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
 Êlayer_regularization_losses
Ëlayer_metrics
Utrainable_variables
V	variables
Wregularization_losses
Ìmetrics
Ínon_trainable_variables
Îlayers
¶__call__
+µ&call_and_return_all_conditional_losses
'µ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
 Ïlayer_regularization_losses
Ðlayer_metrics
Ytrainable_variables
Z	variables
[regularization_losses
Ñmetrics
Ònon_trainable_variables
Ólayers
¸__call__
+·&call_and_return_all_conditional_losses
'·"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'2batch_normalization_4/gamma
(:&2batch_normalization_4/beta
1:/ (2!batch_normalization_4/moving_mean
5:3 (2%batch_normalization_4/moving_variance
.
^0
_1"
trackable_list_wrapper
<
^0
_1
`2
a3"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
 Ôlayer_regularization_losses
Õlayer_metrics
btrainable_variables
c	variables
dregularization_losses
Ömetrics
×non_trainable_variables
Ølayers
º__call__
+¹&call_and_return_all_conditional_losses
'¹"call_and_return_conditional_losses"
_generic_user_object
 :2hidden1/kernel
:2hidden1/bias
.
f0
g1"
trackable_list_wrapper
.
f0
g1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
 Ùlayer_regularization_losses
Úlayer_metrics
htrainable_variables
i	variables
jregularization_losses
Ûmetrics
Ünon_trainable_variables
Ýlayers
¼__call__
+»&call_and_return_all_conditional_losses
'»"call_and_return_conditional_losses"
_generic_user_object
 :
2hidden2/kernel
:
2hidden2/bias
.
l0
m1"
trackable_list_wrapper
.
l0
m1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
 Þlayer_regularization_losses
ßlayer_metrics
ntrainable_variables
o	variables
pregularization_losses
àmetrics
ánon_trainable_variables
âlayers
¾__call__
+½&call_and_return_all_conditional_losses
'½"call_and_return_conditional_losses"
_generic_user_object
 :
2hidden3/kernel
:2hidden3/bias
.
r0
s1"
trackable_list_wrapper
.
r0
s1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
 ãlayer_regularization_losses
älayer_metrics
ttrainable_variables
u	variables
vregularization_losses
åmetrics
ænon_trainable_variables
çlayers
À__call__
+¿&call_and_return_all_conditional_losses
'¿"call_and_return_conditional_losses"
_generic_user_object
 :2Network/kernel
:2Network/bias
.
x0
y1"
trackable_list_wrapper
.
x0
y1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
 èlayer_regularization_losses
élayer_metrics
ztrainable_variables
{	variables
|regularization_losses
êmetrics
ënon_trainable_variables
ìlayers
Â__call__
+Á&call_and_return_all_conditional_losses
'Á"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¶
 ílayer_regularization_losses
îlayer_metrics
~trainable_variables
	variables
regularization_losses
ïmetrics
ðnon_trainable_variables
ñlayers
Ä__call__
+Ã&call_and_return_all_conditional_losses
'Ã"call_and_return_conditional_losses"
_generic_user_object
!:2Response/kernel
:2Response/bias
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
 òlayer_regularization_losses
ólayer_metrics
trainable_variables
	variables
regularization_losses
ômetrics
õnon_trainable_variables
ölayers
Æ__call__
+Å&call_and_return_all_conditional_losses
'Å"call_and_return_conditional_losses"
_generic_user_object
:	 (2
Nadam/iter
: (2Nadam/beta_1
: (2Nadam/beta_2
: (2Nadam/decay
: (2Nadam/learning_rate
: (2Nadam/momentum_cache
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
(
÷0"
trackable_list_wrapper
>
`0
a1
2
3"
trackable_list_wrapper
ö
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
`0
a1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
Ø

øtotal

ùcount
ú	variables
û	keras_api"
_tf_keras_metric{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 69}
:  (2total
:  (2count
0
ø0
ù1"
trackable_list_wrapper
.
ú	variables"
_generic_user_object
):'2Nadam/CovEmb/embeddings/m
):'2Nadam/SexEmb/embeddings/m
*:(2Nadam/FuelEmb/embeddings/m
+:)2Nadam/UsageEmb/embeddings/m
+:)2Nadam/FleetEmb/embeddings/m
(:&P2Nadam/PcEmb/embeddings/m
/:-2#Nadam/batch_normalization_4/gamma/m
.:,2"Nadam/batch_normalization_4/beta/m
&:$2Nadam/hidden1/kernel/m
 :2Nadam/hidden1/bias/m
&:$
2Nadam/hidden2/kernel/m
 :
2Nadam/hidden2/bias/m
&:$
2Nadam/hidden3/kernel/m
 :2Nadam/hidden3/bias/m
&:$2Nadam/Network/kernel/m
 :2Nadam/Network/bias/m
):'2Nadam/CovEmb/embeddings/v
):'2Nadam/SexEmb/embeddings/v
*:(2Nadam/FuelEmb/embeddings/v
+:)2Nadam/UsageEmb/embeddings/v
+:)2Nadam/FleetEmb/embeddings/v
(:&P2Nadam/PcEmb/embeddings/v
/:-2#Nadam/batch_normalization_4/gamma/v
.:,2"Nadam/batch_normalization_4/beta/v
&:$2Nadam/hidden1/kernel/v
 :2Nadam/hidden1/bias/v
&:$
2Nadam/hidden2/kernel/v
 :
2Nadam/hidden2/bias/v
&:$
2Nadam/hidden3/kernel/v
 :2Nadam/hidden3/bias/v
&:$2Nadam/Network/kernel/v
 :2Nadam/Network/bias/v
Ö2Ó
B__inference_model_3_layer_call_and_return_conditional_losses_18347
B__inference_model_3_layer_call_and_return_conditional_losses_18467
B__inference_model_3_layer_call_and_return_conditional_losses_18110
B__inference_model_3_layer_call_and_return_conditional_losses_18181À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ó2Ð
 __inference__wrapped_model_17191«
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢

 
Designÿÿÿÿÿÿÿÿÿ
"
Coverageÿÿÿÿÿÿÿÿÿ

Sexÿÿÿÿÿÿÿÿÿ

Fuelÿÿÿÿÿÿÿÿÿ

Usageÿÿÿÿÿÿÿÿÿ

Fleetÿÿÿÿÿÿÿÿÿ
$!

PostalCodeÿÿÿÿÿÿÿÿÿ
 
LogGAMÿÿÿÿÿÿÿÿÿ
ê2ç
'__inference_model_3_layer_call_fn_17659
'__inference_model_3_layer_call_fn_18519
'__inference_model_3_layer_call_fn_18571
'__inference_model_3_layer_call_fn_18039À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ë2è
A__inference_CovEmb_layer_call_and_return_conditional_losses_18580¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ð2Í
&__inference_CovEmb_layer_call_fn_18587¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ë2è
A__inference_SexEmb_layer_call_and_return_conditional_losses_18596¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ð2Í
&__inference_SexEmb_layer_call_fn_18603¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ì2é
B__inference_FuelEmb_layer_call_and_return_conditional_losses_18612¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ñ2Î
'__inference_FuelEmb_layer_call_fn_18619¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_UsageEmb_layer_call_and_return_conditional_losses_18628¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ò2Ï
(__inference_UsageEmb_layer_call_fn_18635¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_FleetEmb_layer_call_and_return_conditional_losses_18644¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ò2Ï
(__inference_FleetEmb_layer_call_fn_18651¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ê2ç
@__inference_PcEmb_layer_call_and_return_conditional_losses_18660¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ï2Ì
%__inference_PcEmb_layer_call_fn_18667¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_Cov_flat_layer_call_and_return_conditional_losses_18673¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ò2Ï
(__inference_Cov_flat_layer_call_fn_18678¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_Sex_flat_layer_call_and_return_conditional_losses_18684¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ò2Ï
(__inference_Sex_flat_layer_call_fn_18689¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_Fuel_flat_layer_call_and_return_conditional_losses_18695¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ó2Ð
)__inference_Fuel_flat_layer_call_fn_18700¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_Usage_flat_layer_call_and_return_conditional_losses_18706¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ô2Ñ
*__inference_Usage_flat_layer_call_fn_18711¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_Fleet_flat_layer_call_and_return_conditional_losses_18717¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ô2Ñ
*__inference_Fleet_flat_layer_call_fn_18722¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ì2é
B__inference_Pc_flat_layer_call_and_return_conditional_losses_18728¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ñ2Î
'__inference_Pc_flat_layer_call_fn_18733¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ì2é
B__inference_concate_layer_call_and_return_conditional_losses_18745¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ñ2Î
'__inference_concate_layer_call_fn_18756¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Þ2Û
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_18776
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_18810´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
¨2¥
5__inference_batch_normalization_4_layer_call_fn_18823
5__inference_batch_normalization_4_layer_call_fn_18836´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ì2é
B__inference_hidden1_layer_call_and_return_conditional_losses_18847¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ñ2Î
'__inference_hidden1_layer_call_fn_18856¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ì2é
B__inference_hidden2_layer_call_and_return_conditional_losses_18867¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ñ2Î
'__inference_hidden2_layer_call_fn_18876¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ì2é
B__inference_hidden3_layer_call_and_return_conditional_losses_18887¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ñ2Î
'__inference_hidden3_layer_call_fn_18896¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ì2é
B__inference_Network_layer_call_and_return_conditional_losses_18906¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ñ2Î
'__inference_Network_layer_call_fn_18915¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
è2å
>__inference_Add_layer_call_and_return_conditional_losses_18921¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Í2Ê
#__inference_Add_layer_call_fn_18927¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_Response_layer_call_and_return_conditional_losses_18938¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ò2Ï
(__inference_Response_layer_call_fn_18947¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
þBû
#__inference_signature_wrapper_18241CoverageDesignFleetFuelLogGAM
PostalCodeSexUsage"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 Æ
>__inference_Add_layer_call_and_return_conditional_losses_18921Z¢W
P¢M
KH
"
inputs/0ÿÿÿÿÿÿÿÿÿ
"
inputs/1ÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
#__inference_Add_layer_call_fn_18927vZ¢W
P¢M
KH
"
inputs/0ÿÿÿÿÿÿÿÿÿ
"
inputs/1ÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¤
A__inference_CovEmb_layer_call_and_return_conditional_losses_18580_#/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 |
&__inference_CovEmb_layer_call_fn_18587R#/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ£
C__inference_Cov_flat_layer_call_and_return_conditional_losses_18673\3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 {
(__inference_Cov_flat_layer_call_fn_18678O3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¦
C__inference_FleetEmb_layer_call_and_return_conditional_losses_18644_7/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 ~
(__inference_FleetEmb_layer_call_fn_18651R7/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¥
E__inference_Fleet_flat_layer_call_and_return_conditional_losses_18717\3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 }
*__inference_Fleet_flat_layer_call_fn_18722O3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¥
B__inference_FuelEmb_layer_call_and_return_conditional_losses_18612_-/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 }
'__inference_FuelEmb_layer_call_fn_18619R-/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¤
D__inference_Fuel_flat_layer_call_and_return_conditional_losses_18695\3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 |
)__inference_Fuel_flat_layer_call_fn_18700O3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¢
B__inference_Network_layer_call_and_return_conditional_losses_18906\xy/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 z
'__inference_Network_layer_call_fn_18915Oxy/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ£
@__inference_PcEmb_layer_call_and_return_conditional_losses_18660_</¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 {
%__inference_PcEmb_layer_call_fn_18667R</¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¢
B__inference_Pc_flat_layer_call_and_return_conditional_losses_18728\3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 z
'__inference_Pc_flat_layer_call_fn_18733O3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¥
C__inference_Response_layer_call_and_return_conditional_losses_18938^/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 }
(__inference_Response_layer_call_fn_18947Q/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¤
A__inference_SexEmb_layer_call_and_return_conditional_losses_18596_(/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 |
&__inference_SexEmb_layer_call_fn_18603R(/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ£
C__inference_Sex_flat_layer_call_and_return_conditional_losses_18684\3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 {
(__inference_Sex_flat_layer_call_fn_18689O3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¦
C__inference_UsageEmb_layer_call_and_return_conditional_losses_18628_2/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 ~
(__inference_UsageEmb_layer_call_fn_18635R2/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¥
E__inference_Usage_flat_layer_call_and_return_conditional_losses_18706\3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 }
*__inference_Usage_flat_layer_call_fn_18711O3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ
 __inference__wrapped_model_17191ö<72-(#a^`_fglmrsxy¦¢¢
¢

 
Designÿÿÿÿÿÿÿÿÿ
"
Coverageÿÿÿÿÿÿÿÿÿ

Sexÿÿÿÿÿÿÿÿÿ

Fuelÿÿÿÿÿÿÿÿÿ

Usageÿÿÿÿÿÿÿÿÿ

Fleetÿÿÿÿÿÿÿÿÿ
$!

PostalCodeÿÿÿÿÿÿÿÿÿ
 
LogGAMÿÿÿÿÿÿÿÿÿ
ª "3ª0
.
Response"
Responseÿÿÿÿÿÿÿÿÿ¶
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_18776ba^`_3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¶
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_18810b`a^_3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
5__inference_batch_normalization_4_layer_call_fn_18823Ua^`_3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
5__inference_batch_normalization_4_layer_call_fn_18836U`a^_3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ
B__inference_concate_layer_call_and_return_conditional_losses_18745½¢
¢
ü
"
inputs/0ÿÿÿÿÿÿÿÿÿ
"
inputs/1ÿÿÿÿÿÿÿÿÿ
"
inputs/2ÿÿÿÿÿÿÿÿÿ
"
inputs/3ÿÿÿÿÿÿÿÿÿ
"
inputs/4ÿÿÿÿÿÿÿÿÿ
"
inputs/5ÿÿÿÿÿÿÿÿÿ
"
inputs/6ÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ü
'__inference_concate_layer_call_fn_18756°¢
¢
ü
"
inputs/0ÿÿÿÿÿÿÿÿÿ
"
inputs/1ÿÿÿÿÿÿÿÿÿ
"
inputs/2ÿÿÿÿÿÿÿÿÿ
"
inputs/3ÿÿÿÿÿÿÿÿÿ
"
inputs/4ÿÿÿÿÿÿÿÿÿ
"
inputs/5ÿÿÿÿÿÿÿÿÿ
"
inputs/6ÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¢
B__inference_hidden1_layer_call_and_return_conditional_losses_18847\fg/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 z
'__inference_hidden1_layer_call_fn_18856Ofg/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¢
B__inference_hidden2_layer_call_and_return_conditional_losses_18867\lm/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 z
'__inference_hidden2_layer_call_fn_18876Olm/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ
¢
B__inference_hidden3_layer_call_and_return_conditional_losses_18887\rs/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ

ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 z
'__inference_hidden3_layer_call_fn_18896Ors/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ

ª "ÿÿÿÿÿÿÿÿÿ·
B__inference_model_3_layer_call_and_return_conditional_losses_18110ð<72-(#a^`_fglmrsxy®¢ª
¢¢

 
Designÿÿÿÿÿÿÿÿÿ
"
Coverageÿÿÿÿÿÿÿÿÿ

Sexÿÿÿÿÿÿÿÿÿ

Fuelÿÿÿÿÿÿÿÿÿ

Usageÿÿÿÿÿÿÿÿÿ

Fleetÿÿÿÿÿÿÿÿÿ
$!

PostalCodeÿÿÿÿÿÿÿÿÿ
 
LogGAMÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ·
B__inference_model_3_layer_call_and_return_conditional_losses_18181ð<72-(#`a^_fglmrsxy®¢ª
¢¢

 
Designÿÿÿÿÿÿÿÿÿ
"
Coverageÿÿÿÿÿÿÿÿÿ

Sexÿÿÿÿÿÿÿÿÿ

Fuelÿÿÿÿÿÿÿÿÿ

Usageÿÿÿÿÿÿÿÿÿ

Fleetÿÿÿÿÿÿÿÿÿ
$!

PostalCodeÿÿÿÿÿÿÿÿÿ
 
LogGAMÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 È
B__inference_model_3_layer_call_and_return_conditional_losses_18347<72-(#a^`_fglmrsxy¿¢»
³¢¯
¤ 
"
inputs/0ÿÿÿÿÿÿÿÿÿ
"
inputs/1ÿÿÿÿÿÿÿÿÿ
"
inputs/2ÿÿÿÿÿÿÿÿÿ
"
inputs/3ÿÿÿÿÿÿÿÿÿ
"
inputs/4ÿÿÿÿÿÿÿÿÿ
"
inputs/5ÿÿÿÿÿÿÿÿÿ
"
inputs/6ÿÿÿÿÿÿÿÿÿ
"
inputs/7ÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 È
B__inference_model_3_layer_call_and_return_conditional_losses_18467<72-(#`a^_fglmrsxy¿¢»
³¢¯
¤ 
"
inputs/0ÿÿÿÿÿÿÿÿÿ
"
inputs/1ÿÿÿÿÿÿÿÿÿ
"
inputs/2ÿÿÿÿÿÿÿÿÿ
"
inputs/3ÿÿÿÿÿÿÿÿÿ
"
inputs/4ÿÿÿÿÿÿÿÿÿ
"
inputs/5ÿÿÿÿÿÿÿÿÿ
"
inputs/6ÿÿÿÿÿÿÿÿÿ
"
inputs/7ÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
'__inference_model_3_layer_call_fn_17659ã<72-(#a^`_fglmrsxy®¢ª
¢¢

 
Designÿÿÿÿÿÿÿÿÿ
"
Coverageÿÿÿÿÿÿÿÿÿ

Sexÿÿÿÿÿÿÿÿÿ

Fuelÿÿÿÿÿÿÿÿÿ

Usageÿÿÿÿÿÿÿÿÿ

Fleetÿÿÿÿÿÿÿÿÿ
$!

PostalCodeÿÿÿÿÿÿÿÿÿ
 
LogGAMÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
'__inference_model_3_layer_call_fn_18039ã<72-(#`a^_fglmrsxy®¢ª
¢¢

 
Designÿÿÿÿÿÿÿÿÿ
"
Coverageÿÿÿÿÿÿÿÿÿ

Sexÿÿÿÿÿÿÿÿÿ

Fuelÿÿÿÿÿÿÿÿÿ

Usageÿÿÿÿÿÿÿÿÿ

Fleetÿÿÿÿÿÿÿÿÿ
$!

PostalCodeÿÿÿÿÿÿÿÿÿ
 
LogGAMÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ 
'__inference_model_3_layer_call_fn_18519ô<72-(#a^`_fglmrsxy¿¢»
³¢¯
¤ 
"
inputs/0ÿÿÿÿÿÿÿÿÿ
"
inputs/1ÿÿÿÿÿÿÿÿÿ
"
inputs/2ÿÿÿÿÿÿÿÿÿ
"
inputs/3ÿÿÿÿÿÿÿÿÿ
"
inputs/4ÿÿÿÿÿÿÿÿÿ
"
inputs/5ÿÿÿÿÿÿÿÿÿ
"
inputs/6ÿÿÿÿÿÿÿÿÿ
"
inputs/7ÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ 
'__inference_model_3_layer_call_fn_18571ô<72-(#`a^_fglmrsxy¿¢»
³¢¯
¤ 
"
inputs/0ÿÿÿÿÿÿÿÿÿ
"
inputs/1ÿÿÿÿÿÿÿÿÿ
"
inputs/2ÿÿÿÿÿÿÿÿÿ
"
inputs/3ÿÿÿÿÿÿÿÿÿ
"
inputs/4ÿÿÿÿÿÿÿÿÿ
"
inputs/5ÿÿÿÿÿÿÿÿÿ
"
inputs/6ÿÿÿÿÿÿÿÿÿ
"
inputs/7ÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿæ
#__inference_signature_wrapper_18241¾<72-(#a^`_fglmrsxyî¢ê
¢ 
âªÞ
.
Coverage"
Coverageÿÿÿÿÿÿÿÿÿ
*
Design 
Designÿÿÿÿÿÿÿÿÿ
(
Fleet
Fleetÿÿÿÿÿÿÿÿÿ
&
Fuel
Fuelÿÿÿÿÿÿÿÿÿ
*
LogGAM 
LogGAMÿÿÿÿÿÿÿÿÿ
2

PostalCode$!

PostalCodeÿÿÿÿÿÿÿÿÿ
$
Sex
Sexÿÿÿÿÿÿÿÿÿ
(
Usage
Usageÿÿÿÿÿÿÿÿÿ"3ª0
.
Response"
Responseÿÿÿÿÿÿÿÿÿ