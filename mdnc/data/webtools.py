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

import tqdm

__all__ = ['get_token',
           'download_tarball_link', 'download_tarball_public', 'download_tarball_private', 'download_tarball',
           'DataChecker']


class _SafePoolManager(urllib3.PoolManager):
    '''A wrapped urllib3.PoolManager with context supported.
    This is a private class. Should not be used by users.
    '''
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.clear()


def get_token(token='', silent=False):
    '''Automatically get the token, if the token is missing.
    Arguments:
        token: the given OAuth token. Only when this argument is unset,
               the program will try to find a token from env.
        silent: a flag. If set true, this tool would not ask for a token
                when the token could not be found.
    '''
    if not token:
        token = os.environ.get('GITTOKEN', None)
        if token is None:
            token = os.environ.get('GITHUB_API_TOKEN', None)
        if isinstance(token, str) and token != '':
            token = token.split(':')[-1]
        else:
            if not silent:
                print('data.webtools: A Github OAuth token is required for downloading the data in private repository. Please provide your OAuth token:')
                token = input('Token:')
                if not token:
                    print('data.webtools: Provide blank token. Try to download the tarball without token.')
                print('data.webtools: Tips: specify the environment variable $GITTOKEN or $GITHUB_API_TOKEN could help you skip this step.')
            else:
                return ''
    return token


def __get_tarball_mode(name, mode='auto'):
    '''Detect the tarball compression mode by file name.
    Arguments:
        name: the file name with a file name extension.
        mode: the mode name, should be '', 'gz, ''bz2', or 'xz'. If specified,
              the compression mode would not be detected by file name.
    '''
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


def download_tarball_link(link, path='.', mode='auto', verbose=False):
    '''Download an online tarball and extract it automatically.
    The tarball is directed by the link. This tool would not work on
    private github repository.
    The tarball would be sent to pipeline and not get stored.
    Now supports gz, bz2 or xz format.
    Arguments:
        link: the web link.
        path: the extracted data root path. Should be a folder path.
        mode: the mode of extraction. Could be 'gz', 'bz2', 'xz' or
              'auto'.
        verbose: a flag, whether to show the downloaded size during
                 the web request.
    '''
    mode = __get_tarball_mode(name=link, mode=mode)
    os.makedirs(path, exist_ok=True)
    # Initialize urllib3
    with _SafePoolManager(retries=urllib3.util.Retry(connect=5, read=2, redirect=5),
                          timeout=urllib3.util.Timeout(connect=5.0)) as http:
        # Get the data.
        git_header = {
            'User-Agent': 'cainmagi/webtools'
        }
        req = http.request(url=link, headers=git_header, method='GET', preload_content=False)
        if req.status < 400:
            if verbose:
                file_name = os.path.split(link)[-1]
                with tqdm.tqdm.wrapattr(req, 'read', total=0, desc='Get {0}'.format(file_name)) as req:
                    with tarfile.open(fileobj=req, mode='r|{0}'.format(mode)) as tar:
                        tar.extractall(path)
            else:
                with tarfile.open(fileobj=req, mode='r|{0}'.format(mode)) as tar:
                    tar.extractall(path)
        else:
            raise FileNotFoundError('data.webtools: Fail to get access to the tarball. Maybe the repo or the tag is not correct, or the repo is private, or the network is not available. The error message is: {0}'.format(req.read().decode('utf-8')))
        req.release_conn()


def __download_tarball_from_repo(user, repo, tag, asset, path='.', mode='auto', token=None, verbose=False):
    '''Download an online tarball and extract it automatically.
    A base tool. Should not used by users. Please use
        download_tarball, or
        download_tarball_public, or
        download_tarball_private
    for instead.
    '''
    # Initialize the urllib3
    with _SafePoolManager(retries=urllib3.util.Retry(connect=5, read=2, redirect=5),
                          timeout=urllib3.util.Timeout(connect=5.0)) as http:
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
            if verbose:
                with tqdm.tqdm.wrapattr(req, 'read', total=0, desc='Get {0}'.format(asset)) as req:
                    with tarfile.open(fileobj=req, mode='r|{0}'.format(mode)) as tar:
                        tar.extractall(path)
            else:
                with tarfile.open(fileobj=req, mode='r|{0}'.format(mode)) as tar:
                    tar.extractall(path)
        else:
            raise FileNotFoundError('data.webtools: Fail to get access to the asset. The error message is: {0}'.format(req.read().decode('utf-8')))
        req.release_conn()


def download_tarball_public(user, repo, tag, asset, path='.', mode='auto', verbose=False):
    '''Download an online tarball and extract it automatically
    (public).
    This tool only supports public github repositories. This method
    could be replaced by download_tarball_link(), but we do not
    recommend to do that.
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
        verbose: a flag, whether to show the downloaded size during
                 the web request.
    '''
    mode = __get_tarball_mode(name=asset, mode=mode)
    os.makedirs(path, exist_ok=True)
    __download_tarball_from_repo(user=user, repo=repo, tag=tag, asset=asset,
                                 path=path, mode=mode, token=None, verbose=verbose)


def download_tarball_private(user, repo, tag, asset, path='.', mode='auto', token=None, verbose=False):
    '''Download an online tarball and extract it automatically
    (private).
    This tool should only be used for downloading assets from
    private repositories. Although it could be also used for
    public repositories, we do not recommend to use it in those
    cases, because it would still require a token.
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
        verbose: a flag, whether to show the downloaded size during
                 the web request.
    '''
    mode = __get_tarball_mode(name=asset, mode=mode)
    os.makedirs(path, exist_ok=True)
    token = get_token(token)
    __download_tarball_from_repo(user=user, repo=repo, tag=tag, asset=asset,
                                 path=path, mode=mode, token=token, verbose=verbose)


def download_tarball(user, repo, tag, asset, path='.', mode='auto', token=None, verbose=False):
    '''Download an online tarball and extract it automatically.
    This tool is used for downloading the assets from github
    repositories. It would try to detect the data info in public
    mode, and switch to private downloading mode when the Github
    repository could not be accessed.
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
        token: the token required for downloading the private asset,
               when downloading public asses, this value would not
               be used.
        verbose: a flag, whether to show the downloaded size during
                 the web request.
    '''
    mode = __get_tarball_mode(name=asset, mode=mode)
    os.makedirs(path, exist_ok=True)
    # Detect the repository infomation first.
    is_public_mode = True
    with _SafePoolManager(retries=urllib3.util.Retry(connect=5, read=2, redirect=5),
                          timeout=urllib3.util.Timeout(connect=5.0)) as http:
        link_full = 'https://api.github.com/repos/{user}/{repo}/releases/tags/{tag}'.format(user=user, repo=repo, tag=tag)
        git_header = {
            'Accept': 'application/vnd.github.v3+json',
            'User-Agent': 'cainmagi/webtools'
        }
        req = http.request(url=link_full, headers=git_header, method='GET', preload_content=False)
        if req.status < 400:
            is_public_mode = is_public_mode
        else:
            is_public_mode = False
        req.release_conn()
    if is_public_mode:
        __download_tarball_from_repo(user=user, repo=repo, tag=tag, asset=asset,
                                     path=path, mode=mode, token=None, verbose=verbose)
    else:
        token = get_token(token)
        __download_tarball_from_repo(user=user, repo=repo, tag=tag, asset=asset,
                                     path=path, mode=mode, token=token, verbose=verbose)


class DataChecker:
    '''Check the existence of the required datasets.
    This data checker could check the local dataset folder, find the not existing
    datasets and fetch those required datasets from online repositories or links.
    A private repository requires a token.
    '''

    def __init__(self, root='./datasets', set_list_file='web-data', token='', verbose=False):
        '''Initialization
        Arguments:
            root: the root path of all maintained local datasets.
            set_list_file: a json file recording the online repository paths of the
                           required datasets.
            token: the default Github OAuth token for downloading files from private
                   repositories. If not set, the downloading from public
                   repositories would not be influenced.
            verbose: a flag, whether to show the downloaded size during
                     the web request.
        '''
        set_list_file = os.path.splitext(set_list_file)[0] + '.json'
        with open(set_list_file, 'r') as f:
            self.set_list = json.load(f)
        self.query_list = list()
        self.root = root
        self.token = get_token(token, silent=True)
        self.verbose = verbose

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
                    },
                    {
                        'tag': 'test',
                        'asset': 'test-datasets-2.tar.xz',
                        'items': [
                            'test_data_h5converter.h5',
                            'test_data_h5cparser.h5',
                            'test_data_h5cparser_seq.h5',
                            'test_data_h5gparser.h5',
                            'test_data_h5seqconverter1.h5',
                            'test_data_h5seqconverter2.h5'
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

    @staticmethod
    def __query_with_ext(set_folder, set_name, query_list):
        set_name, set_ext = os.path.splitext(set_name)
        trails = [set_name + set_ext, ]
        if set_ext == '':
            trails.append(set_name + '.h5')
            trails.append(set_name + '.nc')
            trails.append(set_name + '.bcolz')
        require = False
        for tr in trails:
            if tr in query_list:
                require = True
                break
        if require:
            for tr in trails:
                if os.path.isfile(os.path.join(set_folder, tr)):
                    return False
            return True
        return False

    def query(self):
        '''Search the files in the query list, and download the datasets.'''
        query_list = set(self.query_list)
        set_folder = os.path.join(self.root)
        required_sets = list()
        for dinfo in self.set_list['set_list']:
            for set_name in dinfo['items']:
                if self.__query_with_ext(set_folder, set_name, query_list):
                    required_sets.append(dinfo)
                    break
        if required_sets:
            print('data.webtools: There are required dataset missing. Start downloading from the online repository...')
            user = self.set_list.get('user', 'cainmagi')
            repo = self.set_list.get('repo', 'MDNC')
            for reqset in required_sets:
                download_tarball(user=user, repo=repo,
                                 tag=reqset['tag'], asset=reqset['asset'], path=set_folder,
                                 token=self.token, verbose=self.verbose)
            print('data.webtools: Successfully download all required datasets.')
        else:
            print('data.webtools: All required datasets are available.')
