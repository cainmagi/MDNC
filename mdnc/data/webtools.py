#!python
# -*- coding: UTF-8 -*-
'''
################################################################
# WebTools - sequence
# @ Modern Deep Network Toolkits for pyTorch
# Yuchen Jin @ cainmagi@gmail.com
# Requirements: (Pay attention to version)
#   python 3.3+
#   urllib3 1.26.2+
# Tools used for checking and downloading datasets.
# Inspired by:
#   https://gist.github.com/devhero/8ae2229d9ea1a59003ced4587c9cb236
#   and https://gist.github.com/maxim/6e15aa45ba010ab030c4
################################################################
'''

import os
import json
import tarfile
import urllib3


def get_token(token=''):
    '''Automatically get the token, if the token is missing.'''
    if not token:
        token = os.environ.get('GITTOKEN', None)
        if token is None:
            token = os.environ.get('GITHUB_API_TOKEN', None)
        if token is not None:
            token = token.split(':')[-1]
        else:
            print('data.webtools: A Github OAuth token is required for downloading the data in private repository. Please provide your OAuth token:')
            token = input('Token:')
            if not token:
                print('data.webtools: Provide blank token. Try to download the tarball without token.')
            print('data.webtools: Tips: specify the environment variable $GITTOKEN or $GITHUB_API_TOKEN could help you skip this step.')
    return token


def get_tarball_mode(name, mode='auto'):
    '''Detect the tarball compression mode by file name.'''
    name = os.path.split(name)[-1]
    pos = name.find('?')
    if pos > 0:
        name = name[:name.find('?')]  # Remove the HTML args.
    if mode == 'auto':
        if name.endswith('tar'):
            mode = ''
        elif name.endswith('tar.gz') or name.endswith('tar.gzip'):
            mode = 'gz'
        elif name.endswith('tar.bz2') or name.endswith('tar.bzip2'):
            mode = 'bz2'
        elif name.endswith('tar.xz'):
            mode = 'xz'
    if mode not in ('', 'gz', 'bz2', 'xz'):
        raise TypeError('data.webtools: The file name to be downloaded should end with supported format. Now we supports: tar, tar.gz/tar.gzip, tar.bz2/tar.bzip2, tar.xz.')
    return mode


def download_tarball(link, path='.', mode='auto'):
    '''Download an online tarball and extract it automatically.
    The tarball would be sent to pipeline and not get stored.
    Now supports gz or xz format.
    Arguments:
        link: the web link.
        path: the extracted data root path. Should be a folder path.
        mode: the mode of extraction. Could be 'gz', 'bz2', 'xz' or
              'auto'.
    '''
    mode = get_tarball_mode(name=link, mode=mode)
    os.makedirs(path, exist_ok=True)
    # Initialize urllib3
    http = urllib3.PoolManager(
        retries=urllib3.util.Retry(connect=5, read=2, redirect=5),
        timeout=urllib3.util.Timeout(connect=5.0)
    )
    # Get the data.
    git_header = {
        'User-Agent': 'cainmagi/webtools'
    }
    req = http.request(url=link, headers=git_header, method='GET', preload_content=False)
    if req.status < 400:
        with tarfile.open(fileobj=req, mode='r|{0}'.format(mode)) as tar:
            tar.extractall(path)
    else:
        raise FileNotFoundError('data.webtools: Fail to get access to the tarball. Maybe the repo or the tag is not correct, or the repo is private, or the network is not available. The error message is: {0}'.format(req.read().decode('utf-8')))
    req.release_conn()


def download_tarball_private(user, repo, tag, asset, path='.', mode='auto', token=None):
    '''Download an online tarball and extract it automatically.
    The tarball would be sent to pipeline and not get stored.
    Now supports gz or xz format.
    Arguments:
        user:  the github user name.
        repo:  the github repository name.
        tag:   the github release tag.
        asset: the github asset (tarball) to be downloaded.
        path: the extracted data root path. Should be a folder path.
        mode: the mode of extraction. Could be 'gz', 'bz2', 'xz' or
              'auto'.
        token: the token required for downloading the private asset.
    '''
    mode = get_tarball_mode(name=asset, mode=mode)
    os.makedirs(path, exist_ok=True)
    token = get_token(token)
    # Initialize the urllib3
    http = urllib3.PoolManager(
        retries=urllib3.util.Retry(connect=5, read=2, redirect=5),
        timeout=urllib3.util.Timeout(connect=5.0)
    )
    # Get the release info.
    link_full = 'https://api.github.com/repos/{user}/{repo}/releases/tags/{tag}'.format(user=user, repo=repo, tag=tag)
    git_header = {
        'Accept': 'application/vnd.github.v3+json',
        'User-Agent': 'cainmagi/webtools'
    }
    if token:
        git_header['Authorization'] = 'token {token}'.format(token=token)
    req = http.request(url=link_full, headers=git_header, method='GET', preload_content=False)
    if req.status < 400:
        info = json.load(req)
        link_assets = info['assets_url']
    else:
        raise FileNotFoundError('data.webtools: Fail to get access to the release. Maybe the repo or the tag is not correct, or the authentication fails, or the network is not available. The error message is: {0}'.format(req.read().decode('utf-8')))
    req.release_conn()
    # Get the assets info.
    req = http.request(url=link_assets, headers=git_header, method='GET', preload_content=False)
    if req.status < 400:
        info = json.load(req)
        asset_info = next(filter(lambda aitem: aitem['name'] == asset, info), None)
        if asset_info is None:
            raise FileNotFoundError('data.webtools: Fail to locate the asset "{asset}" in the given release.'.format(asset=asset))
        link_asset = asset_info['url']
    else:
        raise FileNotFoundError('data.webtools: Fail to get access to the release. Maybe the asset address is not correct. The error message is: {0}'.format(req.read().decode('utf-8')))
    req.release_conn()
    # Download the data.
    git_header = {
        'Accept': 'application/octet-stream',
        'User-Agent': 'cainmagi/webtools'
    }
    if token:
        git_header['Authorization'] = 'token {token}'.format(token=token)
    # req = http.request(method='GET', url=link_asset, headers=git_header)
    req = http.request(url=link_asset, headers=git_header, method='GET', preload_content=False)
    if req.status < 400:
        with tarfile.open(fileobj=req, mode='r|{0}'.format(mode)) as tar:
            tar.extractall(path)
    else:
        raise FileNotFoundError('data.webtools: Fail to get access to the asset. The error message is: {0}'.format(req.read().decode('utf-8')))
    req.release_conn()


class DataChecker:
    '''Check the existence of the required datasets.
    This data checker could check the local dataset folder, find the not existing
    datasets and fetch those required datasets from online repositories or links.
    A private repository requires a token.
    '''

    def __init__(self, root='./datasets', set_list_file='web-data', token=''):
        '''Initialization
        Arguments:
            root: the root path of all maintained local datasets.
            set_list_file: a json file recording the online repository paths of the
                           required datasets.
            token: the default Github OAuth token for downloading files from private
                   repositories.
        '''
        set_list_file = os.path.splitext(set_list_file)[0] + '.json'
        with open(set_list_file, 'r') as f:
            self.set_list = json.load(f)
        self.query_list = list()
        self.root = root
        self.token = token

    @staticmethod
    def init_set_list(file_name='web-data'):
        '''Create an example of the set list file.
        This method should get used by users manually.
        Arguments:
            file_name: the name of the created set list file.
        '''
        file_name = os.path.splitext(file_name)[0] + '.json'
        with open(file_name, 'w') as f:
            json.dump(fp=f, obj={
                'set_list': [
                    {
                        'tag': 'test',
                        'asset': 'test-datasets-1.tar.xz',
                        'items': [
                            'dataset_file_name_01.txt',
                            'dataset_file_name_02.txt'
                        ]
                    }
                ],
                'user': 'cainmagi',
                'repo': 'MDNC'
            }, indent=2)

    def clear(self):
        '''Clear the query list.'''
        self.query_list.clear()

    def add_query_file(self, file_names):
        '''Add one or more file names in the query list.
        Add file names into the required dataset name list. For each different application,
        the required datasets could be different. The query file list should be a sub-set
        of the whole list given by "set_list_file".
        Arguments:
            file_names: could be one or a list of file name strs, including all requried
                        dataset names for the current program.
        '''
        if isinstance(file_names, (list, tuple)):
            self.query_list.extend(filter(lambda x: isinstance(x, str), file_names))
        elif isinstance(file_names, str):
            self.query_list.append(file_names)
        else:
            raise TypeError('data.webtools: The argument "file_names" requires to be a str or a sequence.')

    def query(self):
        '''Search the files in the query list, and download the datasets.'''
        query_list = set(self.query_list)
        set_folder = os.path.join(self.root)
        required_sets = list()
        for dinfo in self.set_list['set_list']:
            for set_name in dinfo['items']:
                set_name, set_ext = os.path.splitext(set_name)
                if set_ext == '':
                    set_ext = '.h5'
                set_name = set_name + set_ext
                if set_name in query_list and (not os.path.isfile(os.path.join(set_folder, set_name))):
                    required_sets.append(dinfo)
                    break
        if required_sets:
            print('data.webtools: There are required dataset missing. Start downloading from the online repository...')
            token = get_token(self.token)
            user = self.set_list.get('user', 'cainmagi')
            repo = self.set_list.get('repo', 'Dockerfiles')
            for reqset in required_sets:
                download_tarball_private(user=user, repo=repo,
                                         tag=reqset['tag'], asset=reqset['asset'], path=set_folder,
                                         token=token)
            print('data.webtools: Successfully download all required datasets.')
        else:
            print('data.webtools: All required datasets are available.')
