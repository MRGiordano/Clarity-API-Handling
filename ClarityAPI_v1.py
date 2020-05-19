# -*- coding: utf-8 -*-
"""
@author: Mike R Giordano
github.com/MRGiordano
"""
#----------
#Example Work Flow
#----------
# # All_nodes = clarity_api_download(getdata=True)    #download everything to dictionary
# # All_node_sections = {}
# # DiurnalDfs = {}
# # Nodes_list = clarity_api_getnodecodes()           # get separate list of nodes for later
# # clarity_plantowercorrect(All_nodes)               # perform corrections on pm2.5mass data
# # Nodes_stats = clarity_nodescomparison(All_nodes,colofinterest='pm2_5ConcMass.raw',alsograph=True) #calc mean,std for each node and plot
# # Nodes_r2 = clarity_calcr2(Nodes_stats,Nodes_list)   #calc r2 for the above calcs and save to separate df.
# # for key in All_nodes:
# #     tempdf = All_nodes[key]
# #     ts = clarity_isotimetodatetime(tempdf,tdeltafromUTC=3)
# #     splitweekslist = [datetime.datetime(2020,3,23,tzinfo=ts[0].tzinfo),
# #                   datetime.datetime(2020,3,30,tzinfo=ts[0].tzinfo),
# #                   datetime.datetime(2020,4,6,tzinfo=ts[0].tzinfo),
# #                   datetime.datetime(2020,4,13,tzinfo=ts[0].tzinfo),
# #                   datetime.datetime(2020,4,20,tzinfo=ts[0].tzinfo)]
# #     for j in range(len(splitweekslist)):
# #         node_section = splitdf_bydate(tempdf,timeseries = ts,splittype='section',splitdates=[splitweekslist[j],splitweekslist[j+1]])
# #         All_node_sections[key] = node_section
# #         diurn = diurnaldf_from_fulldf(node_section)
# #         DiurnalDfs[key] = diurn

import requests
import pandas as pd
from pandas.io.json import json_normalize
import dateutil.parser
import pytz
import numpy as np

##----------------------
# Direct API Functions
##----------------------

def clarity_api_makeheader(apikey):
    '''
    Function to make header dictionary for calls to Clarity API
    
    Parameters
    ----------
    apikey : str
       Key for your Clarity API account
       
    Returns
    -------
    dictionary containing apikey and gzip encoding
    '''
    headers = {
        'x-api-key':apikey,
        'Accept-Encoding':'gzip'
        }
    return headers

def clarity_api_makeparams(instcode="", starttime="",endtime="",average="hour",skip=0,limit=20000):
    '''
    Function to make the parameters dictionary for calls to Clarity API. Technically all are optional...
    
    Parameters
    ----------
    instcode : str
        The long device ID(s) to filter, use comma separated string to query multiple devices.
        Technically optional but recommended for use.
    starttime : str
        The start time point of the measurements to filter. Date string is expected to be in ISO 8601 format.
        Example: startTime=2019-01-01T00:00:00Z
        Technically optional but recommended in most cases.
    endtime : str, optional
        The end time point of the measurements to filter. Date string is expected to be in ISO 8601 format.
        The default is "".
    average : str, optional
        The averaging period of the measurements to filter. hour, day are supported. The unit of averaging period is 1.
        The default is "hour".
    skip : int, optional
        The number of records to skip. The default is 0.
    limit : int, optional
        The maximum of the measurements to be returned. The default is 20000 to match gzip encoding.

    Raises
    ------
    ValueError
        if any inputs are not acceptable values.

    Returns
    -------
    params : dict
        dictionary to pass to Clarity API call.

    '''
    if limit < 1:
        raise ValueError('limit has a minimum of 1, pick higher')
    if average not in ('', 'hour', 'day'):
        raise ValueError('average must be hour, day, or \'\'')
    if skip < 0:
        raise ValueError('skip has a minimum of 0, pick higher')
    
    params = {
            'startTime':starttime,
            'limit':str(limit)
            }
    
    if instcode is not clarity_api_makeparams.__defaults__[0]:
        params['code'] = instcode
    
    if endtime is not clarity_api_makeparams.__defaults__[2]:
        params['endTime'] = endtime
    
    if average is not clarity_api_makeparams.__defaults__[3]:
        params['average'] = average
        
    if skip is not clarity_api_makeparams.__defaults__[4]:
        params['skip'] = str(skip)
        
    return params

def clarity_api_urldict(key):
    '''
    Storage dictionary for the possible URLs to call for Clarity API
    
    Parameters
    ----------
    key : str

    Returns
    -------
    str of url requested
    '''
    url  = {
        'base':'https://clarity-data-api.clarity.io/v1',
        'measurements':'https://clarity-data-api.clarity.io/v1/measurements',
        'devices':'https://clarity-data-api.clarity.io/v1/devices',
        'readings':'https://clarity-data-api.clarity.io/v1/readings'
        }
    if key in url:
        return url[key]
    else:
        return -1
        return -1

def clarity_api_checkstatus(apikey):
    '''
    Function to check if Clarity API server is up. Expects a response of '200', else raises error.
    '''
    #make sure api server is up
    apiresponse = requests.get(clarity_api_urldict('base'),headers = clarity_api_makeheader(apikey))
    
    if apiresponse.status_code != 200:
        raise ValueError("Not a 200 response from API")
        return apiresponse.status_code
    else:
        return apiresponse.status_code

def clarity_api_getnodecodes(apikey, getstarttimes=False):
    '''
    Function to get a dataframe of all nodes that are available to a given API key.
    
    Parameters
    ----------
    apikey : str
    
    getstarttimes : bool
        Optional parameter to find when a node was first turned on. Appends to dataframe. Default is False.
    '''
    deviceurl = clarity_api_urldict('devices')
    headers = clarity_api_makeheader(apikey)
    
    response = requests.get(deviceurl,headers=headers)
    Node_df = pd.read_json(response.text)
    
    Node_codes = Node_df['code'].tolist()
    
    if getstarttimes:
        Node_start = Node_df['workingStartAt'].tolist()    
        return Node_codes, Node_start    
    else:
        return Node_codes

def clarity_api_checkmaxdl(headers, params,nodecode,starttime, limit=20000):
    '''
    Sometimes the Clarity API server has latency issues and requesting too many 
    
    Parameters
    ----------
    apikey : str
    
    nodecode : str
        Only check 1 node due to download time.
    starttime : str
        ISO 8601 format datestr.  Example: startTime=2019-01-01T00:00:00Z
    limit : int, optional
        Choose if you ever want less than the API maximum. The default is 20000.

    Returns
    -------
    Returns 1 if limit is ok or returns limit that is ok to the server.
    '''
    url = clarity_api_urldict('measurements')
    response = requests.get(url,params=params,headers=headers) #check default limit of 20k first
    
    try:
        tempdf = pd.read_json(response.text)
        return 1
    except: # on failed download, response.text throws a single line error, should make less general incase other fail codes exist...
        dlfailed = True
        while dlfailed:
            limit = round(limit/2)
            params = clarity_api_makeparams(nodecode,starttime)
            response = requests.get(url,params=params,headers=headers)
            try:
                tempdf = pd.read_json(response.text)
                dlfailed = False
            except:
                dlfailed = True
        return limit

    return -1

def clarity_api_JSONresponsetoDF(url,params,headers,desireddata):
    '''
    Takes a response from the Clarity API and turns the JSON into a dataframe.
    
    desireddata : str
        Can be 'location','characteristics', or 'full'.
        'location' gives a dataframe with long, lat coords
        'characteristics' gives a dataframe with only measurements
        'full' gives combined dataframe
    '''
    if desireddata not in ('location','characteristics','full'):
        raise ValueError('Only requests for location and characteristics are supported')
    response = requests.get(url,params=params,headers=headers)
    tempdf = pd.read_json(response.text)
    
    if desireddata != 'full':
        templist = tempdf[desireddata].tolist()
        datadf = json_normalize(templist)
    else:
        datadf = tempdf
    
    return datadf
        
def clarity_api_download(apikey, nodeslist=(), returndict=True, getdata=False,getlocations=False, writetocsv=False, timezone='', overwriteparams={}):    
    '''
    Actually download from the API and return dict's or a dataframe.
    
    Parameters
    ----------
    apikey : str
    
    nodeslist : list, optional
        List of the nodes of interest. Defaults to all nodes available with API key.
    returndict : bool, optional
        Implemented for debugging purposes only. The default is True.
    getdata : bool, optional
        Get the data records for nodes of interest. The default is False.
    getlocations : bool, optional
        Get the long, lat coords for nodes of interest. The default is False.
    writetocsv : bool, optional
        Output each dataframe to a csv. The default is False.
    timezone : str, optional
        Can be used to fix the auto-UTC setting of the Clarity API. Should be in the Continent/City format.
    overwriteparams : dict, optional
        Allows user to set download parameters manually. Useful to download date ranges. No protections to ensure valid entry.

    Raises
    ------
    ValueError
        
    Returns
    -------
    dict

    '''
    status = clarity_api_checkstatus(apikey)
    if not status ==200:
        raise ValueError('Clarity API may be down.')
    
    
    if nodeslist is not clarity_api_download.__defaults__[0]:
        Node_codes = nodeslist
        Node_start = ['1970-01-01T00:00:00Z']*len(Node_codes)
    else:
        Node_codes, Node_start = clarity_api_getnodecodes(apikey,getstarttimes=True)
        
    Nodes_dict = {}
    Locations_dict = {}
    boo = True
        
    measurementsurl = clarity_api_urldict('measurements')
    headers = clarity_api_makeheader(apikey)
    limitparams = clarity_api_makeparams(instcode=Node_codes[0],starttime=Node_start[0],average='')
    
    limit = clarity_api_checkmaxdl(headers, limitparams,Node_codes[0],Node_start[0])
    if limit == 1:
        limit = 20000
    
    for i in range(3):   #for debug
        limit = 500
    # for i in range(len(Node_codes)):
        downloadparams = clarity_api_makeparams(Node_codes[i], Node_start[i],limit=limit)
        
        if overwriteparams is not clarity_api_download.__defaults__[6]:
            #note there is no protection against poorly formatted params
            downloadparams = overwriteparams
    
        if getlocations:
                locationsdf = clarity_api_JSONresponsetoDF(measurementsurl,downloadparams,headers,'location')
                Locations_dict[Node_codes[i]] = locationsdf
                if writetocsv:
                    csvtitle = Node_codes[i]+'.csv'
                    locationsdf.to_csv(csvtitle)
        
        if getdata:
            while boo:
                characteristicsdf = clarity_api_JSONresponsetoDF(measurementsurl,downloadparams,headers,'characteristics')
                fulldf = clarity_api_JSONresponsetoDF(measurementsurl,downloadparams,headers,'full')
                
                if Node_codes[i] in Nodes_dict: #only matters if limit to download is < total required # of responses, eg API server is lagging
                    t_series = pd.to_datetime(fulldf['time'])
                    
                    if timezone is not clarity_api_download.__defaults__[5]:
                        if timezone in pytz.all_timezones:
                            t_series = t_series.dt.tz_convert(timezone)
                        else:
                            raise ValueError('Invalid timezone string')
                    characteristicsdf.insert(0,'time',t_series)
                    # characteristicsdf.set_index('time',inplace=True)
                    
                    olddf = Nodes_dict[Node_codes[i]]

                    newdf = pd.merge(olddf, characteristicsdf, how='outer', on='time')
                    # Merge creates _x and _y columns, fix this
                    for j in olddf.columns:
                        col_name = j
                        if col_name+"_x" in newdf.columns:
                            x = col_name+"_x"
                            y = col_name+"_y"
                            newdf[col_name] = newdf[y].fillna(newdf[x])
                            newdf.drop([x, y], 1, inplace=True)
                    # fulldf = pd.concat([characteristicsdf,newdf],ignore_index=False)
                    Nodes_dict[Node_codes[i]] = newdf
                else:
                    t_series = pd.to_datetime(fulldf['time'])
                    
                    if timezone is not clarity_api_download.__defaults__[5]:
                        if timezone in pytz.all_timezones:
                            t_series = t_series.dt.tz_convert(timezone)
                        else:
                            raise ValueError('Invalid timezone string')
                    characteristicsdf.insert(0,'time',t_series)
                    # characteristicsdf.set_index('time',inplace=True)
                    Nodes_dict[Node_codes[i]] = characteristicsdf
        
                if dateutil.parser.isoparse(fulldf['time'][fulldf.shape[0]-1]) <= dateutil.parser.isoparse(Node_start[i]):
                    #make sure you have all the data from the start of the node's deployment
                    boo = False
                else:
                    downloadparams = clarity_api_makeparams(Node_codes[i],starttime = Node_start[i], endtime = fulldf['time'][fulldf.shape[0]-1],limit=limit)
            
            if writetocsv:
                csvtitle = Node_codes[i]+'.csv'
                realdf = Nodes_dict[Node_codes[i]]
                realdf.sort_values('time',axis=0)
                realdf.to_csv(csvtitle)
            boo = True
    
    if returndict:
        if getlocations and getdata:
            return Nodes_dict, Locations_dict
        elif getlocations:
            return Locations_dict
        elif getdata:
            return Nodes_dict
        
##----------------------
# Utility Functions
##----------------------

def datetimetoclarityiso(year,mon,day,hr,mm,ss):
    '''
    Ensures a datetime input is in proper Clarity ISO8601 format. All int inputs.
    '''
    #note that clarity only returns UTC time
    iso8601format = str(year)+'-'+str(mon)+'-'+str(day)+'T'+str(hr)+':'+str(mm)+':'+str(ss)+'Z'
    return iso8601format

def timezonefromlatlong(lat,long):
    client = geonames.GeonamesClient('demo')
    result = client.find_timezone({'lat': lat, 'lng': long})
    return result['timezoneId']

def calcdewpoint(rh,temp):
    '''
    Calculate dewpoint from RH, Temp.

    Parameters
    ----------
    rh : float
        Relative Humidity. Can be % or decimal.
    temp : float
        Temperature. Assumed to be in C if < 200, else K.

    Returns
    -------
    dp : float
        Dew Point.

    '''
    if temp < 200:
        temp = temp+273.15  #convert to K in a dumb way.
    if rh > 1.5:
        rh = rh/100         #get rh out of %
    
    e = rh* (6.112 * np.exp(17.67 * (temp- 273.15)/ (temp - 29.65)))
    loge = np.log(e / 6.112)
    dp= 0. + 243.5 * loge / (17.67 - loge)   
    return dp

##----------------------
# Clarity Dataframe Functions
##----------------------
def clarity_plantowercorrect(Nodes_dict,custom_coeffs=[]):
    '''
    Correct the pm2.5 data utilizing the Plantower correction in Malings et al., 2019. AS&T.
    
    Parameters
    ----------
    Nodes_dict : dict OR dataframe
        dictionary of dataframes (OR single dataframe) that all share a common column to be averaged
    custom_coeffs : list
        Use own coefficients for correction.
    '''
    if custom_coeffs is clarity_plantowercorrect.__defaults__[0]:   # from malings et al 
        beta = [75, 0.6, -2.5, -0.82, 2.9]  #pm2.5 mass > 20 ug/m3
        gamma = [21, 0.43, -0.58, -0.22, 0.73] #pm2.5 mass <= 20 ug/m3
    
    if type(Nodes_dict) is dict:
        for key in Nodes_dict:
            nodetemp = Nodes_dict[key]['temperature.value'].tolist()
            noderh = Nodes_dict[key]['relHumid.value'].tolist()
            dp = [calcdewpoint(noderh[x], nodetemp[x]) for x in range(len(noderh))]
            
            nodemass = Nodes_dict[key]['pm2_5ConcMass.raw'].tolist()
            nodemasscorr = list()
            for i in range(len(nodemass)):
                if(nodemass[i] > 20):
                    nodemasscorr.append(beta[0] + beta[1]*nodemass[i] + beta[2]*nodetemp[i] + beta[3]*noderh[i] + beta[4]*dp[i])
                if(nodemass[i] <= 20):
                    nodemasscorr.append(gamma[0] + gamma[1]*nodemass[i] + gamma[2]*nodetemp[i] + gamma[3]*noderh[i] + gamma[4]*dp[i])
            
            insertloc = Nodes_dict[key].columns.tolist().index('pm2_5ConcMass.raw')+1
            Nodes_dict[key].insert(insertloc, 'pm2_5ConcMass.corr',nodemasscorr)

    if isinstance(Nodes_dict, pd.DataFrame):
        nodetemp = Nodes_dict['temperature.value'].tolist()
        noderh = Nodes_dict['relHumid.value'].tolist()
        dp = [calcdewpoint(noderh[x], nodetemp[x]) for x in range(len(noderh))]
        
        nodemass = Nodes_dict['pm2_5ConcMass.raw'].tolist()
        nodemasscorr = list()
        for i in range(len(nodemass)):
            if(nodemass[i] > 20):
                nodemasscorr.append(beta[0] + beta[1]*nodemass[i] + beta[2]*nodetemp[i] + beta[3]*noderh[i] + beta[4]*dp[i])
            if(nodemass[i] <= 20):
                nodemasscorr.append(gamma[0] + gamma[1]*nodemass[i] + gamma[2]*nodetemp[i] + gamma[3]*noderh[i] + gamma[4]*dp[i])
        
        insertloc = Nodes_dict.columns.tolist().index('pm2_5ConcMass.raw')+1
        Nodes_dict.insert(insertloc, 'pm2_5ConcMass.corr',nodemasscorr)
    
def clarity_calcmeanfromdict(Nodes_dict,coltoavg='pm2_5ConcMass.corr'):
    '''
    Calculate mean of a specific column in a dictionary of dataframes, appends to each dataframe
    
    Parameters
    ----------
    Nodes_dict : dict
        dictionary of dataframes that all share a common column to be averaged
    coltoavg : str
        column to average.
    '''    
    if type(Nodes_dict) != dict:
        raise ValueError('Expected a dictionary (of dataframes)')
    dftoavg = pd.DataFrame()
    for key in Nodes_dict:
        tempdf = Nodes_dict[key]
        try:
            if len(tempdf[coltoavg].columns.tolist()) > 1: #likely given a diurnal (or 'described' df)
                dftoavg[key] = tempdf[coltoavg]['mean']
        except:
            dftoavg[key] = tempdf[coltoavg]
    
    dftoavg['mean'] = dftoavg[list(Nodes_dict.keys())].mean(axis=1,skipna=True,numeric_only=True)
    
    return dftoavg

def clarity_nodescomparison(Nodes_dict, colofinterest='pm2_5ConcMass.value',alsograph=False):
    #input: dictionary containing all nodes of interest full characteristics df's
    #output a single dataframe with columns: time, 1 col for each node in nodes_dict, count,mean,std, sem
    
    #find largest overlap in time among all keys in the dict
    fulltimeslist = []
    datadf = pd.DataFrame()
    for key in Nodes_dict:
        temptimeslist = Nodes_dict[key]['time'].tolist()
        fulltimeslist.extend(x for x in temptimeslist if x not in fulltimeslist)
     
    datadf['time'] = fulltimeslist

    for key in Nodes_dict:
        datadf = datadf[:].merge(Nodes_dict[key][['time',colofinterest]],on='time',how='left')
        datadf = datadf.rename(columns={colofinterest:key})
    
    #add a 'count' column and a mean column - former = mean of all sensors (using keys in dict), latter = mean/sum of same to get number of sensors that contribute to that sum
    datadf['count'] = datadf[list(Nodes_dict.keys())].sum(axis=1,skipna=True,numeric_only=True)/datadf[list(Nodes_dict.keys())].mean(axis=1,skipna=True,numeric_only=True)
    datadf['mean'] = datadf[list(Nodes_dict.keys())].mean(axis=1,skipna=True,numeric_only=True)
    #also add st.dev. and st. err. mean for each measurement
    datadf['std'] = datadf[list(Nodes_dict.keys())].std(axis=1,skipna=True,numeric_only=True)
    datadf['sem'] = datadf[list(Nodes_dict.keys())].sem(axis=1,skipna=True,numeric_only=True)
    
    if alsograph:
        clarity_graph_nodescomparison(datadf, list(Nodes_dict.keys()))
        return datadf
    else:
        return datadf

def clarity_graph_nodescomparison(datadf, nodes_list, axislog=False,semstd='sem'):
    nodescomp_fig, nodescomp_ax = plt.subplots()
    
    colormap = pd.DataFrame()
    #plot each sensor, add the "1:1" line and the SEM
    for i in nodes_list:
        colormap[i] = abs((datadf[i]-datadf['mean'])/datadf['mean'])/datadf[semstd]
        nodescomp_ax.scatter(datadf[i],datadf['mean'],label=i)#,c=colormap[i])
        
    nodescomp_ax.plot(datadf['mean'],datadf['mean'],'k')
    
    # meanplus = datadf['mean'] + datadf[semstd]
    # meanminus = datadf['mean'] - datadf[semstd]
    
    # nodescomp_ax.scatter(datadf['mean'],meanplus,'k')
    # nodescomp_ax.plot(datadf['mean'],meanminus,'g')
    
    nodescomp_ax.grid(True)    
    nodescomp_ax.set_xlabel('Individual')
    nodescomp_ax.set_ylabel('Mean of All')
    plt.legend()
    
    #set axis scales
    if axislog:
        nodescomp_ax.set_xscale('log')
        nodescomp_ax.set_yscale('log')
    axismin = 0.9*min(datadf[nodes_list].min(axis=0,skipna=True,numeric_only=True))
    axismax = 1.1*max(datadf[nodes_list].max(axis=0,skipna=True,numeric_only=True))
    # nodescomp_ax.set_xticks(ticks=np.linspace(axismin-(axismin*0.5),axismax,20) ,minor=True)
    nodescomp_ax.set_xlim(axismin,axismax)
    nodescomp_ax.set_ylim(axismin,axismax)
    
    # return colormap
    
def clarity_calcr2(datadf,nodes_list):
    #input: output of _nodescomparison
    if 'mean' not in datadf.columns:
        raise ValueError('Expected a dataframe with the column "mean"')

    r2 = pd.DataFrame()
    for i in nodes_list:
        temp = datadf.dropna(axis=0,subset=[i,'mean'])
        r2.loc[0,i] = np.corrcoef(temp[i],temp['mean'])[0][1]**2

    return r2

def diurnaldf_from_fulldf(datadf1):
    datadf = datadf1
    datadf['time'] = pd.to_datetime(datadf['time'])
    datadf.set_index('time',inplace=True)
    
    datadf['diurntime'] = datadf.index.map(lambda x: x.strftime("%H:%M"))
    datadf.reset_index(drop=True,inplace=True)
    datadf.set_index('diurntime',inplace=True)
    datadf = datadf.groupby('diurntime').describe()
    
    return datadf

def splitdf_bydate(datadf, timeseries, splittype, splitdates):
    if splittype == 'in 2':
        if not isinstance(splitdates,datetime.datetime):
            return -1
        splitdate = splitdates
        try:
            splitloc = timeseries.index(min(timeseries,key=lambda x:abs(x-splitdate))) #have to do it on temptime because otherwise typeError
            
            if timeseries[splitloc-1] > splitdate:
                datadf_presplit = datadf.iloc[splitloc:datadf.shape[0],:]#everthing after split
                datadf_postsplit = datadf.iloc[0:splitloc,:]#everything before split
            else:
                datadf_postsplit = datadf.iloc[splitloc:datadf.shape[0],:]#everthing after split
                datadf_presplit = datadf.iloc[0:splitloc,:]#everything before split
            
            return datadf_presplit, datadf_postsplit
        except:
            raise ValueError('Date not found in df')
    
    if splittype == 'section':
        if not isinstance(splitdates,list) or not isinstance(splitdates[0],datetime.datetime) or not isinstance(splitdates[1],datetime.datetime):
            raise ValueError('Splitdates needs to be in datetime.datetime format')
        
        try:
            splitstart = timeseries.index(min(timeseries,key=lambda x:abs(x-splitdates[0])))
            splitend = timeseries.index(min(timeseries,key=lambda x:abs(x-splitdates[1])))
            
            if splitstart < splitend:
                datadf_section = datadf.iloc[splitstart:splitend,:]
            else:
                datadf_section = datadf.iloc[splitend:splitstart,:]
            
            return datadf_section
        except:
            raise ValueError('Something went wrong trying to section the df')
    return -1
